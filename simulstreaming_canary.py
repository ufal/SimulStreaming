#!/usr/bin/env python3

import copy
import glob
import json
import os
import argparse
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Optional

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

from whisper_streaming.base import OnlineProcessorInterface
import logging

logger = logging.getLogger(__name__)

def make_divisible_by(num, factor: int) -> int:
    """Make num divisible by factor"""
    return (num // factor) * factor

@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for buffered inference.
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

    # Chunked configs
    chunk_secs: float = 1  # Chunk length in seconds
    left_context_secs: float = (
        10.0  # left context: larger value improves quality without affecting theoretical latency
    )
    right_context_secs: float = 0.5  # right context

    matmul_precision: str = 'high'

    # Decoding strategy for RNNT models
    decoding: AEDStreamingDecodingConfig = field(default_factory=AEDStreamingDecodingConfig)

    # extra arguments for Canary prompt generation
    timestamps: bool = False
    prompt: dict = field(default_factory=dict)

    # debug mode
    debug_mode: bool = False

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

    if cfg.decoding.streaming_policy not in {"alignatt", "waitk"}:
        raise ValueError("This script currently supports only `alignatt` or `waitk` streaming policy")
    
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
            multitask_decoding.strategy = "greedy"
            self.model.change_decoding_strategy(multitask_decoding)

        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        self.model.eval()

        # sampling and encoder parameters
        self.audio_sample_rate = self.model_cfg.preprocessor.sample_rate
        self.feature_stride_sec = self.model_cfg.preprocessor.window_stride
        self.encoder_subsampling_factor = self.model.encoder.subsampling_factor

        # compute context frames and context_samples
        features_per_sec = 1.0 / self.feature_stride_sec
        features_frame2audio_samples = make_divisible_by(
            int(self.audio_sample_rate * self.feature_stride_sec), factor=self.encoder_subsampling_factor
        )
        self.encoder_frame2audio_samples = features_frame2audio_samples * self.encoder_subsampling_factor
        
        self.context_encoder_frames = ContextSize(
            left=int(self.cfg.left_context_secs * features_per_sec / self.encoder_subsampling_factor),
            chunk=int(self.cfg.chunk_secs * features_per_sec / self.encoder_subsampling_factor),
            right=int(self.cfg.right_context_secs * features_per_sec / self.encoder_subsampling_factor),
        )

        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left * self.encoder_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * self.encoder_frame2audio_samples,
            right=self.context_encoder_frames.right * self.encoder_frame2audio_samples,
        )

        # decoding computer
        self.decoding_computer = GreedyBatchedStreamingAEDComputer(
            self.model,
            frame_chunk_size=self.context_encoder_frames.chunk,
            decoding_cfg=self.cfg.decoding,
        )

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
        self.expected_samples = asr.context_samples.chunk + asr.context_samples.right
        self.audio_sample_rate = asr.audio_sample_rate
        self.end_of_window_sample = asr.context_samples.chunk
        self._init_stream_state()

    def init(self):
        self._init_stream_state()

    def _init_stream_state(self):
        self.audio_chunks = []
        self.is_last = False
        self.offset = 0.0

        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=self.cfg.batch_size,
            context_samples=self.asr.context_samples,
            dtype=torch.float32,
            device=self.device,
        )

        self.model_state = GreedyBatchedStreamingAEDComputer.initialize_aed_model_state(
            asr_model=self.model,
            decoder_input_ids=self.asr.decoder_input_ids,
            batch_size=self.cfg.batch_size,
            context_encoder_frames=self.asr.context_encoder_frames,
            chunk_secs=self.cfg.chunk_secs,
            right_context_secs=self.cfg.right_context_secs,
        )

    def insert_audio_chunk(self, audio):
        if audio is None:
            return
        
        t = torch.from_numpy(audio).float().to(self.device).clone()
        self.audio_chunks.append(t)

    def _concat_audio_chunks(self):
        if not self.audio_chunks:
            return None
        
        return torch.cat(self.audio_chunks, dim=0)

    def process_iter(self):
        audio_tensor = self._concat_audio_chunks()
        # reset audio_chunks accumulator
        self.audio_chunks = []

        if audio_tensor is None or audio_tensor.numel() == 0:
            return {}

        num_samples = audio_tensor.shape[0]
        seconds = float(num_samples) / float(self.audio_sample_rate)
        self.offset += seconds

        audio_batch = audio_tensor.unsqueeze(0)
        audio_lengths = torch.tensor([audio_batch.shape[1]], device=self.device)

        is_last_chunk = self.is_last
        is_last_chunk_batch = torch.tensor([is_last_chunk], device=self.device)

        self.buffer.add_audio_batch_(
            audio_batch,
            audio_lengths=audio_lengths,
            is_last_chunk=is_last_chunk,
            is_last_chunk_batch=is_last_chunk_batch,
        )

        self.model_state.is_last_chunk_batch = is_last_chunk_batch

        with torch.inference_mode():
             # get encoder output using full buffer [left-chunk-right]
            _, encoded_len, enc_states, _ = self.model(
                input_signal=self.buffer.samples,
                input_signal_length=self.buffer.context_size_batch.total(),
            )

            # remove right context from encoder output length (only for non-last chunks)
            encoder_context_batch = self.buffer.context_size_batch.subsample(
                factor=self.asr.encoder_subsampling_factor
            )

            encoded_len_no_rc = encoder_context_batch.left + encoder_context_batch.chunk
            encoded_length_corrected = torch.where(
                is_last_chunk_batch,
                encoded_len,
                encoded_len_no_rc,
            )
            encoder_input_mask = lens_to_mask(encoded_length_corrected, enc_states.shape[1]).to(enc_states.dtype)

            self.model_state.prev_encoder_shift = max(
                0,
                (self.end_of_window_sample // self.asr.encoder_frame2audio_samples)
                - self.asr.context_encoder_frames.left
                - self.asr.context_encoder_frames.chunk,
            )

            # decoding step
            self.model_state = self.asr.decoding_computer(
                encoder_output=enc_states,
                encoder_output_len=encoded_length_corrected,
                encoder_input_mask=encoder_input_mask,
                prev_batched_state=self.model_state,
            )

        if self.expected_samples > self.asr.context_samples.chunk:
            self.expected_samples -= self.asr.context_samples.right

        # extract predicted tokens for this single stream
        pred_ids = self.model_state.pred_tokens_ids[0, self.model_state.decoder_input_ids.size(-1):self.model_state.current_context_lengths[0]]
        token_list = pred_ids.tolist()
        text = self.model.tokenizer.ids_to_text(token_list).strip()

        if len(text) == 0:
            return {}

        end = self.offset
        start = max(0.0, end - seconds)

        return {
            "start": start,
            "end": end,
            "text": text,
            "tokens": token_list,
        }

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()

        self._init_stream_state()
        return o