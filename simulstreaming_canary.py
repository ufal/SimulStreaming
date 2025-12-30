#!/usr/bin/env python3

import copy
import glob
import json
import os
import argparse
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
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
from nemo.utils import logging as nemo_logging

from whisper_streaming.base import OnlineProcessorInterface
import logging

logger = logging.getLogger(__name__)
nemo_logging.setLevel(logging.INFO)

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
    # TODO: imlement
    pass 

class SimulCanaryOnline(OnlineProcessorInterface):
    # TODO: imlement
    pass 
