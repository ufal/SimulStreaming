from __future__ import annotations

import asyncio
import logging
import os

import torch

from whisper_worker.worker import WhisperWorker, WorkerConfig


def build_config() -> WorkerConfig:
    return WorkerConfig(
        nats_url=os.getenv("NATS_URL", "nats://127.0.0.1:4222"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        opus_subject=os.getenv("OPUS_SUBJECT", "opus.segments.*"),
        subtitles_subject=os.getenv("SUBTITLES_SUBJECT", "subtitles.*"),
        stream_events_subject=os.getenv("STREAM_EVENTS_SUBJECT", "streams.*"),
        lock_bucket=os.getenv("WHISPER_LOCK_BUCKET", "WHISPER_STREAM_LOCKS"),
        lock_ttl_sec=float(os.getenv("WHISPER_LOCK_TTL_SEC", "15")),
        lock_renew_sec=float(os.getenv("WHISPER_LOCK_RENEW_SEC", "5")),
        idle_stream_ttl_sec=float(os.getenv("WHISPER_IDLE_STREAM_TTL_SEC", "10")),
        raw_queue_size=int(os.getenv("WHISPER_RAW_QUEUE_SIZE", "1000")),
        language=os.getenv("WHISPER_LANGUAGE", "ru"),
        task=os.getenv("WHISPER_TASK", "transcribe"),
        model_path=os.getenv(
            "WHISPER_MODEL_PATH", "/home/alex/models/large-v3-turbo.pt"
        ),
        cif_ckpt_path=os.getenv("WHISPER_CIF_CKPT_PATH"),
        beams=int(os.getenv("WHISPER_BEAMS", "1")),
        decoder_type=os.getenv("WHISPER_DECODER", "greedy"),
        frame_threshold=int(os.getenv("WHISPER_FRAME_THRESHOLD", "25")),
        audio_max_len=float(os.getenv("WHISPER_AUDIO_MAX_LEN", "30.0")),
        audio_min_len=float(os.getenv("WHISPER_AUDIO_MIN_LEN", "0.0")),
        min_chunk_size=float(os.getenv("WHISPER_MIN_CHUNK_SIZE", "1.2")),
        never_fire=os.getenv("WHISPER_NEVER_FIRE", "false").lower() == "true",
        init_prompt=os.getenv("WHISPER_INIT_PROMPT"),
        static_init_prompt=os.getenv("WHISPER_STATIC_INIT_PROMPT"),
        max_context_tokens=(
            int(os.getenv("WHISPER_MAX_CONTEXT_TOKENS"))
            if os.getenv("WHISPER_MAX_CONTEXT_TOKENS")
            else None
        ),
        logdir=os.getenv("WHISPER_LOGDIR"),
        warmup_seconds=float(os.getenv("WHISPER_WARMUP_SECONDS", "3")),
        enable_warmup=os.getenv("WHISPER_ENABLE_WARMUP", "false").lower() == "true",
        feed_window_sec=float(os.getenv("WHISPER_FEED_WINDOW_SEC", "1.5")),
        feed_min_flush_sec=float(os.getenv("WHISPER_FEED_MIN_FLUSH_SEC", "0.6")),
        feed_timeout_sec=float(os.getenv("WHISPER_FEED_TIMEOUT_SEC", "0.35")),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("simulstreaming").setLevel(logging.WARNING)
    logging.getLogger("simulstreaming.whisper").setLevel(logging.WARNING)
    logging.getLogger("simulstreaming.whisper.simul_whisper").setLevel(logging.WARNING)
    logging.getLogger("simulstreaming.whisper.simul_whisper.simul_whisper").setLevel(
        logging.WARNING
    )
    logging.getLogger("triton").setLevel(logging.WARNING)


def log_torch_device() -> None:
    log = logging.getLogger("whisper-worker")
    log.info(
        "torch status cuda=%s hip=%s device_count=%d",
        torch.cuda.is_available(),
        getattr(torch.version, "hip", None),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    if torch.cuda.is_available():
        try:
            log.info("torch gpu name=%s", torch.cuda.get_device_name(0))
        except Exception:
            log.info("torch gpu name=unknown")


async def main() -> None:
    cfg = build_config()
    configure_logging(cfg.log_level)
    log_torch_device()
    await WhisperWorker(cfg).run()


if __name__ == "__main__":
    asyncio.run(main())
