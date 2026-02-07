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

from nemo.collections.asr.models.aed_multitask_models import lens_to_mask
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

logger = logging.getLogger(__name__)

@dataclass
class IncrementalOutput:
    new_tokens: List[str]
    new_string: str
    deleted_tokens: List[str]
    deleted_string: str

class PunctuationTextHistory:
    STRONG_PUNCTUATION = [".", "!", "?", ":", ";"]

    def select_text_history(self, text_history):
        new_history = []
        for token in reversed(text_history):
            prefix_token = token
            contains_punctuation = False
            for punct in self.STRONG_PUNCTUATION:
                if punct in prefix_token:
                    contains_punctuation = True
                    break
            if contains_punctuation:
                break
            new_history.append(token)
        # Reverse the list
        return new_history[::-1]

class StreamAttProcessor:
    def __init__(self, asr, stream_cfg, multitask_transcription_conf):
        self.asr = asr
        self.model = asr.model
        self.device = asr.device
        self.cfg = stream_cfg
        self.punct_history = PunctuationTextHistory()
        self.audio_subsampling_factor = self.cfg.audio_subsampling_factor
        self.text_history_max_len = self.cfg.text_history_max_len
        self.cutoff_frame_num = self.cfg.cutoff_frame_num
        self.cross_attn_layer = self.cfg.cross_attn_layer
        self.unselected_tokens = []
        self.audio_history_max_duration = self.cfg.audio_history_max_duration
        self.audio_max_len = self.audio_history_max_duration * 1000 // 10 // self.audio_subsampling_factor

        self.multitask_transcription_conf = multitask_transcription_conf

        #histories
        self.audio_history: Optional[np.ndarray] = None
        self.text_history: List[str] = []
        self.unselected_tokens: List[str] = []

    def _update_text_history(self, new_output: List[str]) -> int:
        if self.text_history:
            self.text_history += new_output
        else:
            self.text_history = new_output
        new_history = self.punct_history.select_text_history(self.text_history)
        discarded_text = len(self.text_history) - len(new_history)
        self.text_history = new_history

        # Ensure not exceeding max text history length
        if self.text_history and len(self.text_history) > self.text_history_max_len:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"The textual history has hit the maximum predefined length of "
                    f"{self.text_history_max_len}")
            self.text_history = self.text_history[-self.text_history_max_len:]
        return discarded_text

    def _cut_audio_exceeding_maxlen(self):
        # Ensure not exceeding max audio history length
        if len(self.audio_history) > self.audio_max_len:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"The audio history has hit the maximum predefined length of "
                    f"{self.audio_max_len}")
            self.audio_history = self.audio_history[-self.audio_max_len:]

    def _update_speech_history(self, discarded_text: int, cross_attn: torch.Tensor) -> None:
        # If no history is discarded, no need for attention-based audio trimming
        if discarded_text == 0:
            # Check audio history not exceeding maximum allowed length
            self._cut_audio_exceeding_maxlen()
            return

        # Trim the cross-attention by excluding the discarded new generated tokens and the
        # discarded textual history. Output shape: (text_history_len, n_audio_features)
        cross_attn = cross_attn[discarded_text:discarded_text + len(self.text_history), :]

        # Compute the frame to which each token of the textual history mostly attends to
        most_attended_idxs = torch.argmax(cross_attn.float(), dim=1)

        # Find the first feature that is attended
        if most_attended_idxs.shape[0] > 1:
            # Multiple tokens: sort and get the earliest attended frame
            sorted_idxs = torch.sort(most_attended_idxs)[0]
            earliest_attended_idx = sorted_idxs[0]
        else:
            # Only one token: use the unique most attended frame
            earliest_attended_idx = most_attended_idxs[0]

        # Multiply by the subsampling factor to recover the original number of frames
        frames_to_cut = earliest_attended_idx * self.audio_subsampling_factor

        # Cut the unattended audio features
        self.audio_history = self.audio_history[frames_to_cut:]

        # Check audio history not exceeding maximum allowed length
        self._cut_audio_exceeding_maxlen()

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

        # Store unselected tokens, to be used in the case of end of stream
        self.unselected_tokens = generated_tokens[len(selected_tokens):]

        return selected_tokens

    def tokens_to_string(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        s = ""
        for tok in tokens:
                s += tok
        return s.strip()

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        return IncrementalOutput(
            new_tokens=generated_tokens,
            new_string=self.tokens_to_string(generated_tokens),
            deleted_tokens=[],
            deleted_string="",
        )

    def process_chunk(self, speech: np.ndarray) -> IncrementalOutput:
        if self.audio_history is not None:
            self.audio_history = np.concatenate((self.audio_history, speech), axis=0)
        else:
            self.audio_history = speech

        new_speech = torch.from_numpy(self.audio_history).to(self.device)
        # Generate new hypothesis with its corresponding cross-attention scores (no prefix)
        generated_tokens, cross_attn = self._generate(new_speech)
        # Select the part of the new hypothesis to be emitted, and trim cross-attention accordingly
        selected_output = self.alignatt_policy(generated_tokens, cross_attn)
        incremental_output = self._build_incremental_outputs(selected_output)
        # Discard textual history, if needed
        discarded_text = self._update_text_history(selected_output)
        # Trim audio corresponding to the discarded textual history
        self._update_speech_history(discarded_text, cross_attn)
        return incremental_output

    def end_of_stream(self) -> IncrementalOutput:
        last_output = self._build_incremental_outputs(self.unselected_tokens)
        self.unselected_tokens = []
        return last_output

    def clear(self) -> None:
        self.text_history = None
        self.audio_history = None
        self.unselected_tokens = []
    
    def _generate(self, speech: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        new_hypothesis = self.model.transcribe(
            speech, 
            override_config=self.multitask_transcription_conf
        )

        return new_hypothesis[0].words, new_hypothesis[0].xatt_scores[self.cross_attn_layer]

@dataclass
class StreamAttConfig:
    """
    Configuration of the StreamAtt policy.
    """
    
    audio_subsampling_factor: int = 1
    cutoff_frame_num: int = 2
    text_history_max_len: int = 128
    cross_attn_layer: int = -2
    audio_history_max_duration: int = 360

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
    stream_cfg = OmegaConf.structured(StreamAttConfig)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both model_path and pretrained_name cannot be None!")
    
    logger.info("Model config:\n%s", OmegaConf.to_yaml(cfg))

    asr = SimulCanaryASR(cfg, stream_cfg)
    return asr, SimulCanaryOnline(asr)

class SimulCanaryASR:
    def __init__(self, cfg: TranscriptionConfig, stream_cfg: StreamAttConfig):
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
            prompt={'source_lang': 'en', 'target_lang': 'en'},
            enable_chunking=self.cfg.enable_chunking,
        )

        self.stream_cfg = stream_cfg

        self.decoder_input_ids = return_decoder_input_ids(self.cfg, self.model)

    def warmup(self, audio):
        """TODO: test. For now not needed"""
        pass

class SimulCanaryOnline(OnlineProcessorInterface):
    def __init__(self, asr: SimulCanaryASR):
        self.asr = asr
        self.model = asr.model
        self.cfg = asr.cfg
        self.device = asr.device
        self.eos_token_id = self.model.tokenizer.eos_id
        self._init_stream_state()
        self.processor = StreamAttProcessor(
            asr, 
            self.asr.stream_cfg,
            self.asr.multitask_transcription_conf    
        )

    def init(self):
        self._init_stream_state()

    def _init_stream_state(self):
        self.audio_chunks = []

    def insert_audio_chunk(self, audio):
        if audio is None:
            return
        
        self.audio_chunks.append(audio)

    def _concat_audio_chunks(self):
        if not self.audio_chunks:
            return None
        
        return np.concatenate(self.audio_chunks, axis=0)

    def process_iter(self):
        speech = self._concat_audio_chunks()
        
        output = self.processor.process_chunk(speech)

        # reset audio_chunks accumulator
        self.audio_chunks = []

        return {
            "start": 0,
            "end": 0,
            "text": output.new_string,
            "tokens": output.new_tokens,
        }

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()

        self._init_stream_state()
        return o