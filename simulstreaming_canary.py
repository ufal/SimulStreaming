#!/usr/bin/env python3

import copy
import glob
import json
import os
import argparse
import csv
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Optional, List, Tuple


import numpy as np
import torch


from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.aed_multitask_models import parse_multitask_prompt
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig
from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig
from simulstreaming.whisper.whisper_streaming.base import OnlineProcessorInterface


import logging

BOW_PREFIX = "\u2581"
CANARY_PRETRAINED_NAME = "nvidia/canary-1b-v2"

logger = logging.getLogger(__name__)

def flatten_list(list):
    flattened =[]
    for chunk in list:
        flattened.extend(chunk)

    return flattened

def load_unboost_words(tsv_path: str, min_percent: float = 0.0) -> List[str]:
    words: List[str] = []
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            word = row["word"]
            try:
                pct = float(row["percent"])
            except (KeyError, ValueError):
                logger.warning("Skipping malformed TSV row: %s", row)
                continue
            if pct >= min_percent:
                words.append(word)
    logger.info(
        "Loaded %d unboost word(s) from %s (min_percent=%.4f)",
        len(words), tsv_path, min_percent,
    )
    return words

@dataclass
class StreamingConfig:
    audio_max_len: int
    frame_threshold: int
    max_context_len: int
    xatt_layer: int = -2 # Layer to take cross attentions from

def simulcanary_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Canary-v2 arguments')
    group.add_argument('--model_path', type=str, default=None, 
                        help='The file path to the Canary .pt model. If not present on the filesystem, the model is downloaded automatically. Does not download twice')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, greedy is used.")
    group.add_argument("--decoder",type=str, choices=["beam", "greedy"], default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=30.0, 
        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--max_context_len', type=float, default=256, 
        help='Max context length in tokens for the decoder')

    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=4, 
        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
            'until this number of encoder frames from the end of audio. In frames: in one second of 16kHz input there is ceil(16000 / 16000*0.01 / 8) = 13 frames.')
    group.add_argument('--strip_incomplete_words', action='store_true', default=False,
        help='If set, trailing incomplete words are stripped '
             'from the AlignAtt output before emission.')

    group = parser.add_argument_group('Prompt and context')
    group.add_argument('--source_lang', type=str, default="en", help='Source language of the input.')
    group.add_argument('--target_lang', type=str, default="en", help='Target language of the output.')
    group.add_argument('--task', type=str, choices=["asr", "ast", "transcribe", "translate", "s2t_translation"], default="transcribe", help='Task')

    group = parser.add_argument_group('Word unboosting (GPU-PB)')
    group.add_argument(
        '--unboost_words_file',
        type=str,
        default=None,
        help=(
            'Path to a TSV file (word<TAB>percent header required)'
        ),
    )
    group.add_argument(
        '--unboost_alpha',
        type=float,
        default=-1.0,
    )
    group.add_argument(
        '--unboost_min_percent',
        type=float,
        default=0.0,
        help=(
            'Only unboost words that appear at least this percentage of the time '
            'in the source transcription TSV. (default: 0.0 = all words in the file)'
        ),
    )
    group.add_argument(
        '--unboost_context_score',
        type=float,
        default=1.0,
        help='GPU-PB context_score for the unboosting tree (default: 1.0).',
    )

def simul_asr_factory(args):
    logger.setLevel(args.log_level)
    decoder = args.decoder

    if args.beams > 1:
            if decoder == "greedy":
                raise ValueError("Invalid 'greedy' decoder type for beams > 1. Use 'beam'.")
            elif decoder is None or decoder == "beam":
                decoder = "beam"
    else:
        if decoder is None:
            decoder = "greedy"
        elif decoder not in ("beam","greedy"):
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")

    if getattr(args, 'unboost_words_file', None) is not None and decoder == "greedy":
        logger.info(
            "Word unboosting requires beam strategy — switching decoder from 'greedy' to 'beam'. "
            "beam_size will remain 1"
        )
        decoder = "beam"
    
    a = { v:getattr(args, v) for v in ["model_path", "decoder", "frame_threshold", "max_context_len", "audio_max_len", "beams", "task",
                                        'source_lang', 'target_lang'
                                       ]}

    a["decoder"] = decoder
    for key in ("unboost_words_file", "unboost_alpha", "unboost_min_percent", "unboost_context_score"):
        a[key] = getattr(args, key, None)

    a["strip_incomplete_words"] = getattr(args, "strip_incomplete_words", False)
    
    asr = SimulCanaryASR(**a)
    return asr, SimulCanaryOnline(asr)

class SimulCanaryASR:
    def __init__(
        self, 
        model_path, 
        decoder, 
        frame_threshold, 
        max_context_len, 
        audio_max_len, 
        beams, 
        task, 
        source_lang, 
        target_lang,
        unboost_words_file: Optional[str] = None,
        unboost_alpha: float = -1.0,
        unboost_min_percent: float = 0.0,
        unboost_context_score: float = 1.0,
        strip_incomplete_words: bool = False,
    ):
        if model_path is not None:
            self.model = ASRModel.restore_from(restore_path=model_path)
        else:
            self.model = ASRModel.from_pretrained(model_name=CANARY_PRETRAINED_NAME)

        self.device = next(self.model.parameters()).device
        self.sample_rate = self.model.cfg.preprocessor.sample_rate

        self.strip_incomplete_words = strip_incomplete_words

        self._unboost_words: List[str] = []
        if unboost_words_file is not None:
            self._unboost_words = load_unboost_words(unboost_words_file, min_percent=unboost_min_percent)

        # Setup decoding strategy
        if hasattr(self.model, 'change_decoding_strategy'):
            multitask_decoding = MultiTaskDecodingConfig()
            multitask_decoding.strategy = decoder
            multitask_decoding.compute_hypothesis_token_set = True
            multitask_decoding.return_xattn_scores = True
            multitask_decoding.beam.beam_size = beams

            if self._unboost_words:
                multitask_decoding.beam.boosting_tree.key_phrases_list = self._unboost_words
                multitask_decoding.beam.boosting_tree.context_score = unboost_context_score
                multitask_decoding.beam.boosting_tree.depth_scaling = 1.0
                multitask_decoding.beam.boosting_tree_alpha = unboost_alpha

            self.model.change_decoding_strategy(multitask_decoding)

        #override default transcribe values with this
        self.multitask_transcription_conf = MultiTaskTranscriptionConfig(
            batch_size=1, # Batch size is one, one input stream per connection
            return_hypotheses=True, # return Hypothesis class
            enable_chunking=False, # Disable chunking because we need cross-attention scores
            timestamps=True,
            verbose=False,
        )

        self.default_prompt = {'source_lang': source_lang, 'target_lang': target_lang, 'task': task}

        self.cfg = StreamingConfig(
            audio_max_len=audio_max_len,
            frame_threshold=frame_threshold,
            max_context_len=max_context_len,
            xatt_layer=self.model.cfg.decoding.get("xatt_scores_layer", -2),
        )

    def warmup(self, audio):
        self.model.transcribe(audio)

    def construct_prompt(self, context, prefix):
        """
        Build input prompt and return decoder_input_ids
        """

        default_turns = self.model.prompt.get_default_dialog_slots()
        default_slots = copy.deepcopy(default_turns[0]["slots"])
        default_slots["decodercontext"] = self.model.tokenizer.ids_to_text(context)
        default_slots["source_lang"] = self.default_prompt['source_lang']
        default_slots["target_lang"] = self.default_prompt['target_lang']
        default_slots["task"] = self.default_prompt['task']

        # Build the turns
        turns = [
            {
                "role": "user", "slots": default_slots
            },
            {
                "role": "user_prefix",
                "slots": {
                    "prefix": f"{self.model.tokenizer.ids_to_text(prefix)}"
                },
            },
        ]

        # The model's output does not work properly when simply setting "turns = turns" in the
        # override_config. So we have to parse the prompt
        prompt_list = parse_multitask_prompt({"turns": turns})

        # Make a copy of the transcription config and set its prompt field
        cfg_copy = copy.deepcopy(self.multitask_transcription_conf)
        cfg_copy.prompt = prompt_list

        encoded = self.model.prompt.encode_dialog(turns=turns)
        decoder_input_ids = encoded["context_ids"].unsqueeze(0).to(self.device)

        return cfg_copy, decoder_input_ids

class SimulCanaryOnline(OnlineProcessorInterface):
    def __init__(self, asr: SimulCanaryASR):
        self.asr = asr
        self.model = asr.model
        self.cfg = asr.cfg
        self._init_stream_state()
        self.sample_rate = asr.sample_rate
        self.strip_incomplete_words = asr.strip_incomplete_words
        self.init()

    def init(self, offset=None):
        self.is_last = False
        self._init_stream_state(offset=offset if offset is not None else 0.0)

    def _init_stream_state(self, offset: float = 0.0):
        self.context_buffer = []
        self.output_history = []
        self.audio_chunks = []
        self.audio_history = []
        self.audio_buffer_offset = offset

    def insert_audio_chunk(self, audio):
        if audio is None:
            return
        
        self.audio_chunks.append(audio)

    def _concat_audio_chunks(self):
        if not self.audio_chunks:
            return None
        
        return np.concatenate(self.audio_chunks, axis=0)

    def _preprocess(self, audio):
        if audio is not None:
            self.audio_history.append(audio)

        return np.concatenate(self.audio_history, axis=0)

    def normalize_attn(self, attn: torch.Tensor):
        """
        Normalize the cross-attention scores along the frame dimension to avoid attention sinks.
        """
        std = attn.std(axis=0)
        std[std == 0.] = 1.0
        mean = attn.mean(axis=0)
        return (attn - mean) / std

    def update_history(self, output: List[int]):
        """
        Update audio history based on the audio_max_len. Push previous output to the output history.
        """

        if output is not None and len(output) > 0:
            self.output_history.append(output)
        else:
            self.output_history.append([])

        total_audio_len = sum(len(audio_chunk) for audio_chunk in self.audio_history)
        while (total_audio_len / self.sample_rate > self.cfg.audio_max_len and
            self.audio_history and
            self.output_history):

            removed_chunk = self.audio_history.pop(0)

            self.audio_buffer_offset += len(removed_chunk) / self.sample_rate
            
            total_audio_len -= len(removed_chunk)
            self.context_buffer.append(self.output_history.pop(0))

        total_context_len = sum(len(chunk) for chunk in self.context_buffer)
        while total_context_len > self.cfg.max_context_len:
            removed = self.context_buffer.pop(0)
            total_context_len -= len(removed)

    def _strip_incomplete_words(self, tokens: List[str]) -> List[str]:
        """
        Remove last incomplete word(s) from the new hypothesis.
        """

        tokens_to_write = []
        # iterate from the end and count how many trailing tokens to drop
        num_tokens_incomplete = 0
        for tok in reversed(tokens):
            num_tokens_incomplete += 1
            if tok.startswith(BOW_PREFIX):
                # slice off the trailing incomplete tokens
                tokens_to_write = tokens[:-num_tokens_incomplete]
                break

        return tokens_to_write

    def alignatt_policy(self, generated_tokens, cross_attn) -> List[str]:
        """
        Apply the AlignAtt policy by cutting off tokens whose attention falls
        beyond the allowed frame range.
        The AlignAtt policy was introduced in:
            S. Papi, et al. 2023. *"AlignAtt: Using Attention-based Audio-Translation
            Alignments as a Guide for Simultaneous Speech Translation"*
            (https://www.isca-archive.org/interspeech_2023/papi23_interspeech.html)
        """
        # Select attention scores corresponding to the new generated tokens
        cross_attn = cross_attn[-len(generated_tokens):, :]
        selected_tokens = generated_tokens

        # Find the frame to which each token mostly attends to
        most_attended_frames = torch.argmax(cross_attn, dim=1)
        cutoff = cross_attn.size(1) - self.cfg.frame_threshold

        # Find the first token that attends beyond the cutoff frame
        invalid_tok_ids = torch.where(most_attended_frames >= cutoff)[0]

        # Truncate tokens up to the first invalid alignment (if any)
        if len(invalid_tok_ids) > 0:
            selected_tokens = selected_tokens[:invalid_tok_ids[0]]

            # Strip incomplete words(if set as a param)
            if self.strip_incomplete_words:
                selected_tokens = self._strip_incomplete_words(selected_tokens)

        return selected_tokens

    def _group_tokens_into_words(
        self,
        token_strings: List[str],
        token_ids: List[int],
    ) -> List[List[int]]:
        """
        Group (token_string, token_id) pairs into words using BOW_PREFIX as a
        word-boundary marker.  Returns a list whose i-th entry is the list of
        token IDs that belong to the i-th word.

        Example
        -------
        token_strings = ["▁Hello", "▁world", "!"]
        → [[id_Hello], [id_world, id_!]]
        """
        words: List[List[int]] = []
        current_ids: List[int] = []

        for tok_str, tok_id in zip(token_strings, token_ids):
            if tok_str.startswith(BOW_PREFIX):
                # A BOW token starts a new word – flush the current one first.
                if current_ids:
                    words.append(current_ids)
                current_ids = [tok_id]
            else:
                # Continuation token (punctuation, sub-word suffix, …)
                current_ids.append(tok_id)

        if current_ids:
            words.append(current_ids)

        return words

    def _build_word_timestamps(
        self,
        token_strings: List[str],
        token_ids: List[int],
        canary_ts_words: List[dict],
    ) -> List[dict]:
        """
        Merge the token grouping with Canary's word-level timestamp list.

        Parameters
        ----------
        token_strings:
            Decoded token strings for the *selected* tokens only.
        token_ids:
            Corresponding token IDs for the selected tokens.
        canary_ts_words:
            The timestamps word list produced by the model for.
        """
        word_token_groups = self._group_tokens_into_words(token_strings, token_ids)

        # The number of words we can emit is limited by the shorter of the two
        # lists: our token grouping may have fewer entries than canary_ts_words
        n_words = min(len(word_token_groups), len(canary_ts_words))

        result: List[dict] = []
        for i in range(n_words):
            ts = canary_ts_words[i]
            result.append({
                'start': ts['start'] + self.audio_buffer_offset,
                'end':   ts['end']   + self.audio_buffer_offset,
                'text':  ts['word'],
                'tokens': word_token_groups[i],
            })

        return result

    def _modify_emit_text(self, tokens: List[str]) -> str:
        #Add space if the first emitted token is not a continuation of the previous
        if len(tokens) > 0 and tokens[0].startswith(BOW_PREFIX):
            return f" {self.model.tokenizer.tokens_to_text(tokens)}"

        return self.model.tokenizer.tokens_to_text(tokens) 

    def process_iter(self):
        speech = self._concat_audio_chunks()
        self.audio_chunks = []

        input_speech = self._preprocess(speech)

        flattened_history = flatten_list(self.output_history)
        flattened_context = flatten_list(self.context_buffer)

        override_config, decoder_input_ids = self.asr.construct_prompt(
            context=flattened_context,
            prefix=flattened_history
        )

        output = self.model.transcribe(
            input_speech,
            override_config=override_config
        )

        generated_tokens = output[0].y_sequence
        if isinstance(generated_tokens, torch.Tensor):
            generated_tokens = generated_tokens.detach().cpu().tolist()
        
        xatt_raw = output[0].xatt_scores[self.cfg.xatt_layer]
        xatt_mean = xatt_raw.mean(dim=0)
        xatt_norm = self.normalize_attn(xatt_mean)

        if self.is_last:
            selected_output = output[0].tokens
        else:
            selected_output = self.alignatt_policy(output[0].tokens, xatt_norm)

        selected_ids: List[int] = generated_tokens[:len(selected_output)]

        #Text to emit
        text = self._modify_emit_text(selected_output)

        self.update_history(selected_ids)

        # Combine timestamps
        canary_ts_words: List[dict] = []
        if output[0].timestamp and 'word' in output[0].timestamp:
                    canary_ts_words = output[0].timestamp['word']

        words = self._build_word_timestamps(
            token_strings=selected_output,
            token_ids=selected_ids,
            canary_ts_words=canary_ts_words,
        )

        if words:
            seg_start = words[0]['start']
            seg_end   = words[-1]['end']
        else:
            seg_start = self.audio_buffer_offset
            seg_end   = self.audio_buffer_offset

        return {
            "start":  seg_start,
            "end":    seg_end,
            "text":   text,
            "tokens": selected_ids,
            "words":  words,
        }

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()

        self.context_history = []
        self.is_last = False

        return o

if __name__ == "__main__":

    from simulstreaming.whisper.whisper_streaming.whisper_online_main import main_simulation_from_file
    main_simulation_from_file(simul_asr_factory, add_args=simulcanary_args)