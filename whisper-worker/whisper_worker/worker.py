from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nats
import numpy as np
from nats.aio.msg import Msg
from nats.js.api import KeyValueConfig, StorageType

from whisper_worker.contracts import decode_audio_envelope, get_headers
from whisper_worker.simulstreaming_engine import (
    SimulStreamingConfig,
    SimulStreamingEngine,
)

log = logging.getLogger("whisper-worker")


def pcm16le_bytes_to_float32(payload: bytes, channels: int = 1) -> np.ndarray:
    if not payload:
        return np.zeros(0, dtype=np.float32)

    if len(payload) % 2 != 0:
        payload = payload[:-1]

    audio_i16 = np.frombuffer(payload, dtype="<i2")

    if channels > 1:
        if audio_i16.size % channels != 0:
            raise ValueError(
                f"pcm payload not aligned to channels={channels}: {audio_i16.size}"
            )
        audio_i16 = (
            audio_i16.reshape(-1, channels).mean(axis=1).astype("<i2", copy=False)
        )

    return (audio_i16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    if src_rate == dst_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    ratio = float(dst_rate) / float(src_rate)
    out_len = max(1, int(round(audio.size * ratio)))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=True, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=True, dtype=np.float32)
    out = np.interp(x_new, x_old, audio.astype(np.float32, copy=False)).astype(
        np.float32
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip()


def fragment_signature(text: str, start: float, end: float, cue_id: str = "") -> str:
    raw = f"{round(start, 3)}|{round(end, 3)}|{text.strip()}|{cue_id}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=10).hexdigest()


def queue_put_latest(queue: asyncio.Queue, item) -> None:
    try:
        queue.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass

    with contextlib.suppress(asyncio.QueueEmpty):
        _ = queue.get_nowait()

    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(item)


def duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


@dataclass(slots=True)
class WorkerConfig:
    nats_url: str = "nats://127.0.0.1:4222"
    log_level: str = "INFO"

    opus_subject: str = "opus.segments.*"
    subtitles_subject: str = "subtitles.*"
    stream_events_subject: str = "streams.*"

    lock_bucket: str = "WHISPER_STREAM_LOCKS"
    lock_ttl_sec: float = 15.0
    lock_renew_sec: float = 5.0
    idle_stream_ttl_sec: float = 10.0

    raw_queue_size: int = 1000

    language: str = "ru"
    task: str = "transcribe"
    model_path: str = "./models/large-v3-turbo.pt"
    cif_ckpt_path: Optional[str] = None
    beams: int = 1
    decoder_type: str = "greedy"
    frame_threshold: int = 25
    audio_max_len: float = 30.0
    audio_min_len: float = 0.0
    min_chunk_size: float = 1.2
    never_fire: bool = False
    init_prompt: Optional[str] = None
    static_init_prompt: Optional[str] = None
    max_context_tokens: Optional[int] = None
    logdir: Optional[str] = None
    warmup_seconds: float = 3.0
    enable_warmup: bool = False

    feed_window_sec: float = 1.5
    feed_min_flush_sec: float = 0.6
    feed_timeout_sec: float = 0.35


@dataclass(slots=True)
class LockState:
    owner: str = ""
    revision: Optional[int] = None
    last_renew: float = 0.0


@dataclass(slots=True)
class PendingItem:
    seq: int
    timestamp_ms: int
    pts: int
    source_capture_unix_ms: Optional[int]
    payload: bytes
    codec: str
    sample_rate: int
    channels: int


@dataclass(slots=True)
class StreamState:
    stream_name: str
    backend: SimulStreamingEngine

    raw_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    decode_task: Optional[asyncio.Task] = None
    renew_task: Optional[asyncio.Task] = None

    closing_requested: bool = False
    finished: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_seen: float = field(default_factory=time.time)
    lock_state: LockState = field(default_factory=LockState)

    pcm_buffer: bytearray = field(default_factory=bytearray)

    published_signatures: set[str] = field(default_factory=set)
    published_fifo: deque[str] = field(default_factory=deque)


class WhisperWorker:
    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self.nc: Optional[nats.NATS] = None
        self.js = None
        self.lock_kv = None
        self.opus_sub = None
        self.events_sub = None
        self.stop_event = asyncio.Event()
        self.dispatch_task: Optional[asyncio.Task] = None
        self.backend: Optional[SimulStreamingEngine] = None

        self.worker_id = (
            f"{os.uname().nodename}:{os.getpid()}:{int(time.time() * 1000)}"
        )
        self.streams: dict[str, StreamState] = {}
        self.incoming: asyncio.Queue[Msg] = asyncio.Queue(maxsize=cfg.raw_queue_size)

    async def build_backend(self) -> SimulStreamingEngine:
        log.info(
            "backend build start model_path=%s language=%s task=%s",
            self.cfg.model_path,
            self.cfg.language,
            self.cfg.task,
        )

        backend = SimulStreamingEngine(
            SimulStreamingConfig(
                language=self.cfg.language,
                task=self.cfg.task,
                model_path=self.cfg.model_path,
                cif_ckpt_path=self.cfg.cif_ckpt_path or "",
                frame_threshold=self.cfg.frame_threshold,
                audio_max_len=self.cfg.audio_max_len,
                audio_min_len=self.cfg.audio_min_len,
                segment_length=self.cfg.min_chunk_size,
                beams=self.cfg.beams,
                decoder_type=self.cfg.decoder_type,
                never_fire=self.cfg.never_fire,
                init_prompt=self.cfg.init_prompt,
                static_init_prompt=self.cfg.static_init_prompt,
                max_context_tokens=self.cfg.max_context_tokens,
                logdir=self.cfg.logdir,
                warmup_seconds=self.cfg.warmup_seconds,
            )
        )

        if self.cfg.enable_warmup:
            warmup_file = Path(__file__).with_name("warmup.mp3")
            if warmup_file.exists():
                try:
                    outputs = await asyncio.wait_for(
                        asyncio.to_thread(backend.warmup_file, warmup_file, 1.5),
                        timeout=3600,
                    )
                    for idx, out in enumerate(outputs, start=1):
                        text = str(out.get("text", ""))
                        log.info("warmup output %d text=%r raw=%s", idx, text, out)
                except Exception:
                    log.exception("warmup failed")

        log.info("backend build done")
        return backend

    async def connect(self) -> None:
        self.nc = nats.NATS()
        await self.nc.connect(servers=[self.cfg.nats_url], name="whisper-worker")
        self.js = self.nc.jetstream()
        await self.ensure_lock_bucket()
        log.info("connected to nats=%s worker_id=%s", self.cfg.nats_url, self.worker_id)

    async def ensure_lock_bucket(self) -> None:
        assert self.js is not None
        try:
            self.lock_kv = await self.js.key_value(self.cfg.lock_bucket)
        except Exception:
            self.lock_kv = await self.js.create_key_value(
                KeyValueConfig(
                    bucket=self.cfg.lock_bucket,
                    history=1,
                    ttl=self.cfg.lock_ttl_sec,
                    storage=StorageType.FILE,
                    replicas=1,
                )
            )

    async def run(self) -> None:
        self.backend = await self.build_backend()
        await self.connect()
        assert self.nc is not None

        self.opus_sub = await self.nc.subscribe(
            self.cfg.opus_subject, cb=self.enqueue_nats_msg
        )
        self.events_sub = await self.nc.subscribe(
            self.cfg.stream_events_subject, cb=self.enqueue_nats_msg
        )

        self.dispatch_task = asyncio.create_task(self.dispatch_loop())

        try:
            while not self.stop_event.is_set():
                await self.cleanup_idle_streams()
                await asyncio.sleep(0.5)
        finally:
            if self.dispatch_task is not None:
                self.dispatch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self.dispatch_task
            await self.shutdown()

    async def enqueue_nats_msg(self, msg: Msg) -> None:
        try:
            self.incoming.put_nowait(msg)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                _ = self.incoming.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.incoming.put_nowait(msg)

    async def dispatch_loop(self) -> None:
        while not self.stop_event.is_set():
            msg = await self.incoming.get()
            try:
                if msg.subject.startswith("opus.segments."):
                    await self.handle_audio(msg)
                elif msg.subject.startswith("streams."):
                    await self.handle_event(msg)
            except Exception:
                log.exception("dispatch failed subject=%s", msg.subject)

    async def handle_event(self, msg: Msg) -> None:
        try:
            stream_name = self.stream_name_from_event(msg)
            if not stream_name:
                return

            if msg.subject.startswith("streams.active."):
                self.ensure_state(stream_name)
                log.info("stream active event stream=%s", stream_name)
            elif msg.subject.startswith("streams.inactive."):
                state = self.streams.get(stream_name)
                if state:
                    state.closing_requested = True
                log.info("stream inactive event stream=%s", stream_name)
        except Exception:
            log.exception("event handler failed")

    async def handle_audio(self, msg: Msg) -> None:
        try:
            envelope = decode_audio_envelope(msg)
        except Exception as exc:
            log.exception("bad audio envelope: %s", exc)
            return

        state = self.ensure_state(envelope.stream_name)
        state.last_seen = time.time()

        if not await self.ensure_owned(state):
            return

        await self.start_stream(state)

        queue_put_latest(
            state.raw_queue,
            PendingItem(
                seq=envelope.seq,
                timestamp_ms=envelope.timestamp_ms,
                pts=envelope.pts,
                source_capture_unix_ms=envelope.source_capture_unix_ms,
                payload=envelope.payload,
                codec=envelope.codec,
                sample_rate=envelope.sample_rate,
                channels=envelope.channels,
            ),
        )

    def ensure_state(self, stream_name: str) -> StreamState:
        state = self.streams.get(stream_name)
        if state is None:
            assert self.backend is not None
            state = StreamState(stream_name=stream_name, backend=self.backend)
            self.streams[stream_name] = state
        return state

    def lock_key(self, stream_name: str) -> str:
        import re

        value = stream_name.strip().lower()
        value = re.sub(r"[^a-z0-9_.-]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("._-")
        return f"streams/{value or 'key'}"

    def lock_payload(self, stream_name: str) -> bytes:
        return json.dumps(
            {"worker_id": self.worker_id, "stream_name": stream_name, "ts": time.time()}
        ).encode("utf-8")

    async def ensure_owned(self, state: StreamState) -> bool:
        async with state.lock:
            if state.closing_requested:
                return False

            if (
                state.lock_state.owner == self.worker_id
                and state.lock_state.revision is not None
            ):
                return True

            key = self.lock_key(state.stream_name)

            try:
                rev = await self.lock_kv.create(
                    key, self.lock_payload(state.stream_name)
                )
                state.lock_state.owner = self.worker_id
                state.lock_state.revision = rev
                state.lock_state.last_renew = time.time()
                log.info("lock acquired stream=%s rev=%s", state.stream_name, rev)
                return True
            except Exception:
                pass

            try:
                entry = await self.lock_kv.get(key)
            except Exception:
                entry = None

            if entry is None or not entry.value:
                return False

            try:
                data = json.loads(entry.value.decode("utf-8"))
            except Exception:
                return False

            if data.get("worker_id") != self.worker_id:
                return False

            state.lock_state.owner = self.worker_id
            state.lock_state.revision = getattr(entry, "revision", None)
            state.lock_state.last_renew = time.time()
            return True

    async def renew_lock(self, state: StreamState) -> bool:
        async with state.lock:
            if (
                state.lock_state.owner != self.worker_id
                or state.lock_state.revision is None
            ):
                return False

            key = self.lock_key(state.stream_name)
            try:
                new_rev = await self.lock_kv.put(
                    key, self.lock_payload(state.stream_name)
                )
                state.lock_state.revision = new_rev
                state.lock_state.last_renew = time.time()
                return True
            except Exception:
                log.warning("lock renew failed stream=%s", state.stream_name)
                state.closing_requested = True
                return False

    async def release_lock(self, state: StreamState) -> None:
        async with state.lock:
            if (
                state.lock_state.owner != self.worker_id
                or state.lock_state.revision is None
            ):
                state.lock_state = LockState()
                return

            with contextlib.suppress(Exception):
                await self.lock_kv.delete(self.lock_key(state.stream_name))
            state.lock_state = LockState()

    async def start_stream(self, state: StreamState) -> None:
        async with state.lock:
            if state.decode_task is None or state.decode_task.done():
                state.decode_task = asyncio.create_task(self.decode_loop(state))
                log.info("decode loop started stream=%s", state.stream_name)

            if state.renew_task is None or state.renew_task.done():
                state.renew_task = asyncio.create_task(self.renew_loop(state))
                log.info("renew loop started stream=%s", state.stream_name)

    async def decode_loop(self, state: StreamState) -> None:
        batch_bytes = int(self.cfg.feed_window_sec * 16000 * 2)
        min_flush_bytes = int(self.cfg.feed_min_flush_sec * 16000 * 2)

        try:
            while not self.stop_event.is_set() and not state.finished:
                chunk = await self._get_next_chunk(state)

                if chunk is not None:
                    pending_batches: list[
                        tuple[bytes, int, int, int, Optional[int]]
                    ] = []

                    async with state.buffer_lock:
                        audio = pcm16le_bytes_to_float32(
                            chunk.payload, channels=chunk.channels
                        )
                        if chunk.sample_rate != 16000:
                            audio = resample_linear(audio, chunk.sample_rate, 16000)

                        if audio.size:
                            state.pcm_buffer.extend(
                                (audio * 32768.0).astype("<i2", copy=False).tobytes()
                            )

                        max_bytes = int(self.cfg.feed_window_sec * 16000 * 2 * 12)
                        if len(state.pcm_buffer) > max_bytes:
                            overflow = len(state.pcm_buffer) - max_bytes
                            del state.pcm_buffer[:overflow]

                        while len(state.pcm_buffer) >= batch_bytes:
                            batch = bytes(state.pcm_buffer[:batch_bytes])
                            del state.pcm_buffer[:batch_bytes]
                            pending_batches.append(
                                (
                                    batch,
                                    chunk.seq,
                                    chunk.pts,
                                    chunk.timestamp_ms,
                                    chunk.source_capture_unix_ms,
                                )
                            )

                    for (
                        batch,
                        seq,
                        pts,
                        timestamp_ms,
                        source_capture_unix_ms,
                    ) in pending_batches:
                        await self.feed_batch(
                            state=state,
                            batch=batch,
                            source_seq=seq,
                            source_pts=pts,
                            source_timestamp_ms=timestamp_ms,
                            source_capture_unix_ms=source_capture_unix_ms,
                        )

                    continue

                if state.closing_requested:
                    async with state.buffer_lock:
                        if len(state.pcm_buffer) >= min_flush_bytes:
                            batch = bytes(state.pcm_buffer)
                            state.pcm_buffer.clear()
                        else:
                            batch = b""

                    if batch:
                        await self.feed_batch(
                            state=state,
                            batch=batch,
                            source_seq=0,
                            source_pts=0,
                            source_timestamp_ms=0,
                            source_capture_unix_ms=None,
                        )
                    break

                if (time.time() - state.last_seen) > self.cfg.idle_stream_ttl_sec:
                    break

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("decode loop failed stream=%s", state.stream_name)
        finally:
            await self.finish_stream(state)

    async def _get_next_chunk(self, state: StreamState) -> Optional[PendingItem]:
        try:
            return await asyncio.wait_for(
                state.raw_queue.get(), timeout=self.cfg.feed_timeout_sec
            )
        except asyncio.TimeoutError:
            return None

    async def feed_batch(
        self,
        *,
        state: StreamState,
        batch: bytes,
        source_seq: int,
        source_pts: int,
        source_timestamp_ms: int,
        source_capture_unix_ms: Optional[int],
    ) -> None:
        if not batch:
            return

        audio = pcm16le_bytes_to_float32(batch, channels=1)
        if audio.size == 0:
            return

        assert self.backend is not None
        batch_start = time.perf_counter()
        outputs = await asyncio.to_thread(self.backend.feed, audio)
        feed_elapsed_ms = duration_ms(batch_start)
        log.info(
            "backend feed done stream=%s batch_sec=%.3f audio_samples=%d outputs=%d elapsed_ms=%.1f",
            state.stream_name,
            len(batch) / (16000.0 * 2.0),
            int(audio.size),
            len(outputs),
            feed_elapsed_ms,
        )

        for out in outputs:
            if not isinstance(out, dict):
                continue

            text = str(out.get("text", ""))
            if not text:
                words = out.get("words") or []
                if isinstance(words, list):
                    parts = []
                    for word in words:
                        if isinstance(word, dict):
                            w = str(word.get("text", ""))
                            if w:
                                parts.append(w)
                    text = " ".join(parts)

            if not text:
                continue

            log.info(
                "recognized fragment stream=%s text=%r start=%.3f end=%.3f final=%s words=%d",
                state.stream_name,
                text,
                float(out.get("start", 0.0)),
                float(out.get("end", 0.0)),
                bool(out.get("is_final", False)),
                len(out.get("words", []) or []),
            )

            await self.publish_fragment(
                state=state,
                result=out,
                text=text,
                source_seq=source_seq,
                source_pts=source_pts,
                source_timestamp_ms=source_timestamp_ms,
                source_capture_unix_ms=source_capture_unix_ms,
            )

    async def publish_fragment(
        self,
        *,
        state: StreamState,
        result: dict,
        text: str,
        source_seq: int,
        source_pts: int,
        source_timestamp_ms: int,
        source_capture_unix_ms: Optional[int],
    ) -> None:
        if self.nc is None or self.backend is None:
            return

        start = float(result.get("start", 0.0))
        end = float(result.get("end", 0.0))
        cue_id = str(result.get("cue_id") or fragment_signature(text, start, end))
        sig = fragment_signature(text, start, end, cue_id)

        if sig in state.published_signatures:
            return

        subtitle = self.backend.result_to_subtitle(
            stream_name=state.stream_name,
            result={
                "text": text,
                "is_final": bool(result.get("is_final", False)),
                "start": start,
                "end": end,
                "emission_time": float(result.get("emission_time", time.time())),
                "cue_id": cue_id,
            },
            source_seq=source_seq,
            source_pts=source_pts,
            source_timestamp_ms=source_timestamp_ms,
            source_capture_unix_ms=source_capture_unix_ms,
        )

        subject = self.caption_subject(state.stream_name)
        headers = {
            "X-Stream-Name": state.stream_name,
            "X-Cue-ID": cue_id,
            "X-Source-Seq": str(source_seq),
            "X-Source-PTS": str(source_pts),
            "X-Source-Timestamp-MS": str(source_timestamp_ms),
        }

        payload = subtitle.to_wire_json()

        publish_start = time.perf_counter()
        await self.nc.publish(subject, payload, headers=headers)
        publish_elapsed_ms = duration_ms(publish_start)
        log.info(
            "subtitle publish done stream=%s mode=nats elapsed_ms=%.1f subject=%s cue_id=%s",
            state.stream_name,
            publish_elapsed_ms,
            subject,
            cue_id,
        )

        state.published_signatures.add(sig)
        state.published_fifo.append(sig)

        while len(state.published_fifo) > 5000:
            old = state.published_fifo.popleft()
            state.published_signatures.discard(old)

        log.info(
            "subtitle published stream=%s text=%r start=%.3f end=%.3f final=%s cue_id=%s",
            state.stream_name,
            text,
            start,
            end,
            bool(result.get("is_final", False)),
            cue_id,
        )

    def caption_subject(self, stream_name: str) -> str:
        base = self.cfg.subtitles_subject.strip()
        key = stream_name.strip().lower()
        key = key.replace(" ", "_")
        import re

        key = re.sub(r"[^a-z0-9_.-]+", "_", key)
        key = re.sub(r"_+", "_", key).strip("._-") or "key"

        if "*" in base:
            return base.replace("*", key)
        if base.endswith("."):
            return base + key
        return f"{base.rstrip('.')}.{key}"

    async def renew_loop(self, state: StreamState) -> None:
        try:
            while not self.stop_event.is_set() and not state.closing_requested:
                await asyncio.sleep(self.cfg.lock_renew_sec)
                if not await self.renew_lock(state):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("renew loop failed stream=%s", state.stream_name)

    async def cleanup_idle_streams(self) -> None:
        for state in list(self.streams.values()):
            if state.finished:
                continue
            if (
                state.decode_task is not None
                and state.decode_task.done()
                and state.raw_queue.empty()
            ):
                await self.finish_stream(state)

    async def finish_stream(self, state: StreamState) -> None:
        if state.finished:
            return

        state.finished = True
        state.closing_requested = True

        async with state.lock:
            tasks = [state.decode_task, state.renew_task]
            state.decode_task = None
            state.renew_task = None

        for task in tasks:
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task

        try:
            assert self.backend is not None
            finish_start = time.perf_counter()
            outputs = await asyncio.to_thread(self.backend.finish)
            finish_elapsed_ms = duration_ms(finish_start)
            log.info(
                "backend finish done stream=%s outputs=%d elapsed_ms=%.1f",
                state.stream_name,
                len(outputs),
                finish_elapsed_ms,
            )
            for out in outputs:
                if isinstance(out, dict):
                    text = str(out.get("text", ""))
                    if not text:
                        continue
                    await self.publish_fragment(
                        state=state,
                        result=out,
                        text=text,
                        source_seq=0,
                        source_pts=0,
                        source_timestamp_ms=0,
                        source_capture_unix_ms=None,
                    )
        except Exception:
            log.exception("finish failed stream=%s", state.stream_name)

        await self.release_lock(state)
        self.streams.pop(state.stream_name, None)
        log.info("stream closed stream=%s", state.stream_name)

    def stream_name_from_event(self, msg: Msg) -> str:
        headers = get_headers(msg)
        if headers:
            v = headers.get("X-Stream-Name", "")
            if v:
                return v
        if msg.subject.startswith("streams.active."):
            return msg.subject.removeprefix("streams.active.")
        if msg.subject.startswith("streams.inactive."):
            return msg.subject.removeprefix("streams.inactive.")
        return ""

    async def shutdown(self) -> None:
        self.stop_event.set()

        states = list(self.streams.values())
        for state in states:
            state.closing_requested = True
            for task in [state.decode_task, state.renew_task]:
                if task is not None:
                    task.cancel()

        for state in states:
            for task in [state.decode_task, state.renew_task]:
                with contextlib.suppress(Exception):
                    if task is not None:
                        await task
            with contextlib.suppress(Exception):
                await self.release_lock(state)

        self.streams.clear()

        if self.nc is not None:
            with contextlib.suppress(Exception):
                await self.nc.drain()
            with contextlib.suppress(Exception):
                await self.nc.close()
