#!/usr/bin/env python3

import copy
import glob
import json
import os
import argparse
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Optional, List, Tuple

import numpy as np

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.models.aed_multitask_models import (
    lens_to_mask,
    parse_multitask_prompt
)
from nemo.collections.asr.parts.submodules.aed_decoding import (
    GreedyBatchedStreamingAEDComputer,
    return_decoder_input_ids,
)
from nemo.collections.asr.parts.submodules.multitask_decoding import (
    AEDStreamingDecodingConfig,
    MultiTaskDecodingConfig,
)
from nemo.collections.asr.parts.utils.streaming_utils import (
    ContextSize,
    StreamingBatchedAudioBuffer,
)
from nemo.collections.asr.parts.utils.transcribe_utils import (
    get_inference_device,
    get_inference_dtype,
    setup_model,
)
from nemo.collections.asr.models.aed_multitask_models import MultiTaskTranscriptionConfig


from whisper_streaming.base import OnlineProcessorInterface
import logging

BOW_PREFIX = "\u2581"

logger = logging.getLogger(__name__)

def flatten_list(list):
    flattened =[]
    for chunk in list:
        flattened.extend(chunk)

    return flattened

@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for canary's inference.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = 'nvidia/canary-1b-v2'  # Name of a pretrained model

    # General configs
    batch_size: int = 1 # Single clint connection

    cuda: Optional[int] = None
    allow_mps: bool = True
    compute_dtype: Optional[str] = (
        None  # "float32", "bfloat16" or "float16"; if None (default): bfloat16 if available, else float32
    )

    matmul_precision: str = 'high'

    # Decoding strategy for RNNT models
    decoding: AEDStreamingDecodingConfig = field(default_factory=AEDStreamingDecodingConfig)

    # extra arguments for Canary prompt generation
    timestamps: bool = False
    prompt: dict = field(default_factory=dict)

    # debug mode
    debug_mode: bool = False

    return_hypotheses: bool = True
    enable_chunking: bool = False

def canary_cfg_options(cls, prefix=""):
    """
    Helper function to get the Canary's available options 
    """

    options = []
    for f in fields(cls):
        name = f"{prefix}{f.name}"

        if is_dataclass(f.type):
            options.extend(canary_cfg_options(f.type, prefix=name + "."))
        else:
            options.append(f"{name} ({f.type.__name__})")
    return options

def simulcanary_args(parser: argparse.ArgumentParser):
    options = canary_cfg_options(TranscriptionConfig)

    parser.add_argument("--canary_cfg", nargs="*", default=[], metavar="KEY=VALUE",
                        help=("List of canary model specific arguments \n:"
                              "Options: \n" + "\n ".join(options)
                              ),
                        )

def simul_asr_factory(args):
    logger.setLevel(args.log_level)

    base_cfg = OmegaConf.structured(TranscriptionConfig)
    overrides = OmegaConf.from_dotlist(args.canary_cfg)
    cfg = OmegaConf.merge(base_cfg, overrides)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both model_path and pretrained_name cannot be None!")
    
    logger.info("Model config:\n%s", OmegaConf.to_yaml(cfg))

    asr = SimulCanaryASR(cfg)
    return asr, SimulCanaryOnline(asr)

class SimulCanaryASR:
    def __init__(self, cfg: TranscriptionConfig):
        self.cfg = cfg

        map_location = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
        compute_dtype = get_inference_dtype(cfg.compute_dtype, device=map_location)
        logger.info(f"Inference will be done on device : {map_location} with compute_dtype: {compute_dtype}")

        # setup ASR model
        self.model, _ = setup_model(cfg, map_location)

        self.device = self.model.device
        self.dtype = get_inference_dtype(cfg.compute_dtype, device=self.device)

        # some changes for streaming scenario
        model_cfg = copy.deepcopy(self.model._cfg)
        with open_dict(model_cfg):
            model_cfg.preprocessor.dither = 0.0
            model_cfg.preprocessor.pad_to = 0

        self.model_cfg = model_cfg

        self.model.freeze()
        self.model.to(self.device)
        self.model.to(self.dtype)

        # Setup decoding strategy
        if hasattr(self.model, 'change_decoding_strategy'):
            multitask_decoding = MultiTaskDecodingConfig()
            multitask_decoding.strategy = "beam"
            multitask_decoding.compute_hypothesis_token_set = True
            multitask_decoding.return_xattn_scores = True
            multitask_decoding.beam.beam_size = 5
            self.model.change_decoding_strategy(multitask_decoding)

        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        self.model.eval()

        #override default transcribe values with this
        self.multitask_transcription_conf = MultiTaskTranscriptionConfig(
            batch_size=self.cfg.batch_size,
            return_hypotheses=self.cfg.return_hypotheses,
            verbose=False,
            enable_chunking=self.cfg.enable_chunking,
        )

    def warmup(self, audio):
        """TODO: test. For now not needed"""
        pass

    def construct_prompt(self, context, prefix):
        """
        Build input prompt and return decoder_input_ids
        """

        default_turns = self.model.prompt.get_default_dialog_slots()
        default_slots = copy.deepcopy(default_turns[0]["slots"])
        default_slots["decodercontext"] = self.model.tokenizer.ids_to_text(context)
        default_slots["source_lang"] = getattr(self.multitask_transcription_conf, "source_lang", "en")
        default_slots["target_lang"] = getattr(self.multitask_transcription_conf, "target_lang", "en")
        default_slots["task"] = getattr(self.multitask_transcription_conf, "task", "asr")

        # Build the prompt
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

        # Make a copy of the transcription config and set its prompt field
        cfg_copy = copy.deepcopy(self.multitask_transcription_conf)
        cfg_copy.turns = turns

        encoded = self.model.prompt.encode_dialog(turns=turns)
        decoder_input_ids = encoded["context_ids"].unsqueeze(0).to(self.device)

        return cfg_copy, decoder_input_ids

class SimulCanaryOnline(OnlineProcessorInterface):
    def __init__(self, asr: SimulCanaryASR):
        self.asr = asr
        self.model = asr.model
        self.cfg = asr.cfg
        self.device = asr.device
        self.eos_token_id = self.model.tokenizer.eos_id
        self._init_stream_state()
        self.xatt_layer = -2 # Layer to take cross-attention from
        self.cutoff_frame_num = 8 # Alignatt frame threshhold
        self.context_buffer = []
        self.output_history = []
        self.max_history_len = 15 # Chunks remembered for the audio
        self.max_context_len = 128 # Maximum len of words in context

    def init(self):
        self._init_stream_state()

    def _init_stream_state(self):
        self.context_buffer = []
        self.output_history = []
        self.audio_chunks = []

    def insert_audio_chunk(self, audio):
        if audio is None:
            return
        
        self.audio_chunks.append(audio)

    def _concat_audio_chunks(self):
        if not self.audio_chunks:
            return None
        
        return np.concatenate(self.audio_chunks, axis=0)

    def normalize_attn(self, attn):
        """
        Normalize the cross-attention scores along the frame dimension to avoid attention sinks.
        """
        std = attn.std(axis=0)
        std[std == 0.] = 1.0
        mean = attn.mean(axis=0)
        return (attn - mean) / std

    def update_history(self, output):
        if output is not None and len(output) > 0:
            self.output_history.append(output)

        if(len(self.audio_chunks) > self.max_history_len):
            self.context_buffer.append(self.output_history.pop(0))
            self.audio_chunks.pop(0)

        total_context_len = sum(len(chunk) for chunk in self.context_buffer)
        while total_context_len > self.max_context_len:
            removed = self.context_buffer.pop(0)
            total_context_len -= len(removed)

    def _strip_incomplete_words(self, tokens: List[str]) -> List[str]:
        """
        Remove last incomplete word(s) from the new hypothesis.

        Args:
            tokens (List[str]): selected tokens, possibly containing partial words to be removed.

        Returns:
            List[str]: A list of generated tokens from which partial words are removed.
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
        cutoff = cross_attn.size(1) - self.cutoff_frame_num

        # Find the first token that attends beyond the cutoff frame
        invalid_tok_ids = torch.where(most_attended_frames >= cutoff)[0]

        # Truncate tokens up to the first invalid alignment (if any)
        if len(invalid_tok_ids) > 0:
            selected_tokens = selected_tokens[:invalid_tok_ids[0]]

        selected_tokens = self._strip_incomplete_words(selected_tokens)

        # Store unselected tokens, to be used in the case of end of stream
        self.unselected_tokens = generated_tokens[len(selected_tokens):]

        return selected_tokens

    def process_iter(self):
        speech = self._concat_audio_chunks()

        flattened_history = flatten_list(self.output_history)
        flattened_context = flatten_list(self.context_buffer)

        override_config, decoder_input_ids = self.asr.construct_prompt(
            context=flattened_context,
            prefix=flattened_history
        )

        output = self.model.transcribe(speech, override_config=override_config)

        generated_tokens = output[0].y_sequence
        if isinstance(generated_tokens, torch.Tensor):
            generated_tokens = generated_tokens.detach().cpu().tolist()
        
        xatt_raw = output[0].xatt_scores[self.xatt_layer][:, :decoder_input_ids.shape[1] + len(generated_tokens), :]
        xatt_mean = xatt_raw.mean(dim=0)
        xatt_norm = self.normalize_attn(xatt_mean)

        selected_output = self.alignatt_policy(generated_tokens, xatt_norm)

        self.update_history(selected_output)

        return {
            "start": 0,
            "end": 0,
            "text": generated_text,
            "tokens": output[0].tokens,
        }

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()

        self._init_stream_state()
        return o