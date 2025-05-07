from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SimulWhisperConfig:
    model_path: str
    language: str = field(default="zh")
    nonspeech_prob: float = 1.0
    min_seg_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)

@dataclass
class AlignAttConfig(SimulWhisperConfig):
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200
    buffer_len: float = 30.0
    if_ckpt_path: str = ""
    never_fire: bool = False