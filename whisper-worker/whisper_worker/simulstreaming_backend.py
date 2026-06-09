from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from whisper_worker.audio_decode import decode_audio_file
from whisper_worker.contracts import CaptionPayload, CaptionWord

log = logging.getLogger("whisper-worker")


@dataclass(slots=True)
class SimulStreamingConfig:
    language: str = "auto"
    task: str = "transcribe"
    model_path: str = "large-v3-turbo.pt"
    cif_ckpt_path: Optional[str] = None
    frame_threshold: int = 25
    audio_max_len: float = 30.0
    audio_min_len: float = 0.0
    segment_length: float = 1.2
    beams: int = 1
    decoder_type: str = "greedy"
    never_fire: bool = False
    init_prompt: Optional[str] = None
    static_init_prompt: Optional[str] = None
    max_context_tokens: Optional[int] = None
    logdir: Optional[str] = None
    warmup_seconds: float = 3.0


class SimulStreamingBackend:
    def __init__(self, cfg: SimulStreamingConfig) -> None:
        self.cfg = cfg
        self.device = self._detect_device()

        log.info(
            "backend init start device=%s torch_cuda=%s torch_hip=%s",
            self.device,
            torch.cuda.is_available(),
            getattr(torch.version, "hip", None),
        )

        from whisper_worker.SimulStreaming.simulstreaming_whisper import (
            SimulWhisperASR,
            SimulWhisperOnline,
        )

        self.asr = SimulWhisperASR(
            language=cfg.language,
            model_path=cfg.model_path,
            cif_ckpt_path=cfg.cif_ckpt_path,
            frame_threshold=cfg.frame_threshold,
            audio_max_len=cfg.audio_max_len,
            audio_min_len=cfg.audio_min_len,
            segment_length=cfg.segment_length,
            beams=cfg.beams,
            task=cfg.task,
            decoder_type=cfg.decoder_type,
            never_fire=cfg.never_fire,
            init_prompt=cfg.init_prompt,
            static_init_prompt=cfg.static_init_prompt,
            max_context_tokens=cfg.max_context_tokens,
            logdir=cfg.logdir,
        )
        self.online = SimulWhisperOnline(self.asr)

        model_device = getattr(getattr(self.asr, "model", None), "device", None)
        log.info("backend init done model_device=%s", model_device)

    def _detect_device(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch does not see CUDA/ROCm. SimulStreaming backend cannot start."
            )
        return "cuda"

    @staticmethod
    def _normalize_result(result: Any) -> list[dict[str, Any]]:
        if not result:
            return []
        if isinstance(result, list):
            return [x for x in result if isinstance(x, dict) and x]
        if isinstance(result, dict):
            return [result]
        return []

    @staticmethod
    def _chunk_audio(audio: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
        if audio.size == 0:
            return []
        chunk_samples = max(1, int(16000 * chunk_seconds))
        return [
            audio[i : i + chunk_samples] for i in range(0, audio.size, chunk_samples)
        ]

    def reset(self) -> None:
        self.online.init()

    def warmup_file(
        self, warmup_path: str | Path, chunk_seconds: float = 1.5
    ) -> list[dict[str, Any]]:
        warmup_path = Path(warmup_path)
        log.info("warmup decode start file=%s", warmup_path)

        audio = decode_audio_file(warmup_path, target_sample_rate=16000).astype(
            np.float32, copy=False
        )
        log.info(
            "warmup decode done samples=%d seconds=%.2f",
            audio.size,
            audio.size / 16000.0,
        )

        if audio.size == 0:
            return []

        self.online.init()
        outputs: list[dict[str, Any]] = []

        chunks = self._chunk_audio(audio, chunk_seconds)
        log.info("warmup chunks=%d chunk_seconds=%.2f", len(chunks), chunk_seconds)

        for idx, chunk in enumerate(chunks, start=1):
            log.info("warmup chunk %d/%d samples=%d", idx, len(chunks), chunk.size)
            self.online.insert_audio_chunk(chunk)
            outputs.extend(self._normalize_result(self.online.process_iter()))

        outputs.extend(self._normalize_result(self.online.finish()))
        self.online.init()
        return outputs

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return []

        self.online.insert_audio_chunk(np.ascontiguousarray(audio))
        return self._normalize_result(self.online.process_iter())

    def finish(self) -> list[dict[str, Any]]:
        result = self._normalize_result(self.online.finish())
        self.online.init()
        return result

    @staticmethod
    def vendor_word_to_caption_word(word: dict[str, Any]) -> CaptionWord:
        return CaptionWord(
            start=float(word.get("start", 0.0)),
            end=float(word.get("end", 0.0)),
            text=str(word.get("text", "")).strip(),
            tokens=[int(t) for t in word.get("tokens", []) if isinstance(t, int)],
        )

    @staticmethod
    def result_to_caption(
        *,
        stream_name: str,
        result: dict[str, Any],
        worker_id: str,
        language: str,
        task: str,
        model_name: str,
        source_seq: int | None = None,
        source_pts: int | None = None,
        source_timestamp_ms: int | None = None,
    ) -> CaptionPayload:
        words = []
        for w in result.get("words", []) or []:
            if isinstance(w, dict) and str(w.get("text", "")).strip():
                words.append(SimulStreamingBackend.vendor_word_to_caption_word(w))

        return CaptionPayload(
            stream_name=stream_name,
            start=float(result.get("start", 0.0)),
            end=float(result.get("end", 0.0)),
            text=str(result.get("text", "")).strip(),
            tokens=[int(t) for t in result.get("tokens", []) if isinstance(t, int)],
            words=words,
            is_final=bool(result.get("is_final", False)),
            emission_time=float(result.get("emission_time", 0.0)),
            model=model_name,
            language=language,
            task=task,
            worker_id=worker_id,
            source_seq=source_seq,
            source_pts=source_pts,
            source_timestamp_ms=source_timestamp_ms,
        )
