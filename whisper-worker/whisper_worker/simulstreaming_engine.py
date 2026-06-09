from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from whisper_worker.audio_decode import decode_audio_file
from whisper_worker.contracts import SubtitlePayload, make_cue_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimulStreamingConfig:
    language: str = "ru"
    model_path: str = "./large-v3-turbo.pt"
    cif_ckpt_path: str = ""
    frame_threshold: int = 25
    audio_max_len: float = 30.0
    audio_min_len: float = 0.0
    segment_length: float = 1.2
    beams: int = 1
    task: str = "transcribe"
    decoder_type: str = "greedy"
    never_fire: bool = False
    init_prompt: Optional[str] = None
    static_init_prompt: Optional[str] = None
    max_context_tokens: Optional[int] = None
    logdir: Optional[str] = None
    warmup_seconds: float = 3.0


class SimulStreamingEngine:
    def __init__(self, cfg: SimulStreamingConfig):
        self.cfg = cfg
        self._asr = None
        self._online = None
        self._ready = False

    def _build(self) -> None:
        from whisper_worker.SimulStreaming.simulstreaming_whisper import (
            SimulWhisperASR,
            SimulWhisperOnline,
        )

        self._asr = SimulWhisperASR(
            language=self.cfg.language,
            model_path=self.cfg.model_path,
            cif_ckpt_path=self.cfg.cif_ckpt_path,
            frame_threshold=self.cfg.frame_threshold,
            audio_max_len=self.cfg.audio_max_len,
            audio_min_len=self.cfg.audio_min_len,
            segment_length=self.cfg.segment_length,
            beams=self.cfg.beams,
            task=self.cfg.task,
            decoder_type=self.cfg.decoder_type,
            never_fire=self.cfg.never_fire,
            init_prompt=self.cfg.init_prompt,
            static_init_prompt=self.cfg.static_init_prompt,
            max_context_tokens=self.cfg.max_context_tokens,
            logdir=self.cfg.logdir,
        )
        self._online = SimulWhisperOnline(self._asr)
        self._ready = True

    def ensure_ready(self) -> None:
        if not self._ready:
            self._build()

    @staticmethod
    def _normalize_result(result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []

        if isinstance(result, dict):
            return [result]

        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict) and item]

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                return []

            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict) and item]

        return []

    def warmup_file(
        self,
        audio_path: str | Path,
        chunk_seconds: float = 1.5,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self._online is not None

        audio = decode_audio_file(audio_path, target_sample_rate=16000)
        if audio.size == 0:
            return []

        chunk_samples = max(1, int(16000 * chunk_seconds))
        outputs: list[dict[str, Any]] = []

        self._online.init()
        for start in range(0, audio.size, chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if chunk.size == 0:
                continue
            self._online.insert_audio_chunk(
                np.ascontiguousarray(chunk, dtype=np.float32)
            )
            outputs.extend(self._normalize_result(self._online.process_iter()))

        outputs.extend(self._normalize_result(self._online.finish()))
        self._online.init()
        return outputs

    def reset(self) -> None:
        if self._online is not None:
            self._online.init()

    @staticmethod
    def _to_float32(audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        return np.ascontiguousarray(audio)

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self._online is not None

        audio = self._to_float32(audio)
        if audio.size == 0:
            return []

        self._online.insert_audio_chunk(audio)
        return self._normalize_result(self._online.process_iter())

    def finish(self) -> list[dict[str, Any]]:
        if not self._ready or self._online is None:
            return []

        result = self._normalize_result(self._online.finish())
        self._online.init()
        return result

    @staticmethod
    def _result_text(result: dict[str, Any]) -> str:
        text = result.get("text")
        if isinstance(text, str) and text != "":
            return text
        if text is not None:
            return str(text)

        words = result.get("words") or []
        if isinstance(words, list) and words:
            parts: list[str] = []
            for word in words:
                if not isinstance(word, dict):
                    continue
                piece = word.get("text", "")
                if piece is None:
                    continue
                parts.append(str(piece))
            return "".join(parts)

        return ""

    @staticmethod
    def result_to_subtitle(
        *,
        stream_name: str,
        result: dict[str, Any],
        source_seq: int | None = None,
        source_pts: int | None = None,
        source_timestamp_ms: int | None = None,
        source_capture_unix_ms: int | None = None,
    ) -> SubtitlePayload:
        text = SimulStreamingEngine._result_text(result)
        start = float(result.get("start", 0.0))
        end = float(result.get("end", 0.0))
        cue_id = str(result.get("cue_id") or make_cue_id(stream_name, text, start, end))
        emission_time = float(result.get("emission_time", 0.0) or 0.0)

        return SubtitlePayload(
            kind="subtitle",
            stream_name=stream_name,
            cue_id=cue_id,
            text=text,
            is_final=bool(result.get("is_final", False)),
            start=start,
            end=end,
            emission_time=emission_time,
            source_seq=source_seq,
            source_pts=source_pts,
            source_timestamp_ms=source_timestamp_ms,
            source_capture_unix_ms=source_capture_unix_ms,
        )

    to_caption = result_to_subtitle
