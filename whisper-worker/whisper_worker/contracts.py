from __future__ import annotations

import base64
import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from nats.aio.msg import Msg

_AUDIO_HEADER = struct.Struct(">HIqI")


@dataclass(slots=True)
class AudioChunkEnvelope:
    stream_name: str
    seq: int
    timestamp_ms: int
    pts: int
    codec: str
    sample_rate: int
    channels: int
    payload: bytes
    source_capture_unix_ms: Optional[int] = None
    received_at: float = field(default_factory=time.time)

    @classmethod
    def from_msg(cls, msg: Msg) -> "AudioChunkEnvelope":
        if not msg.data:
            raise ValueError("empty message body")

        if len(msg.data) >= _AUDIO_HEADER.size:
            try:
                seq, timestamp_ms, pts, payload_len = _AUDIO_HEADER.unpack_from(
                    msg.data, 0
                )
                payload_end = _AUDIO_HEADER.size + payload_len
                if payload_len >= 0 and payload_end == len(msg.data):
                    payload = msg.data[_AUDIO_HEADER.size : payload_end]

                    headers = get_headers(msg)
                    stream_name = headers.get("X-Stream-Name", "")
                    if not stream_name:
                        stream_name = (
                            msg.subject.rsplit(".", 1)[-1]
                            if "." in msg.subject
                            else "default"
                        )

                    capture_header = headers.get("X-Capture-Unix-MS")
                    source_capture_unix_ms = (
                        int(capture_header)
                        if capture_header not in (None, "")
                        else None
                    )

                    return cls(
                        stream_name=stream_name,
                        seq=seq,
                        timestamp_ms=timestamp_ms,
                        pts=pts,
                        codec=headers.get("X-Audio-Codec", "pcm_s16le"),
                        sample_rate=int(headers.get("X-Sample-Rate", "16000")),
                        channels=int(headers.get("X-Channels", "1")),
                        payload=payload,
                        source_capture_unix_ms=source_capture_unix_ms,
                    )
            except Exception:
                pass

        stripped = msg.data.lstrip()
        if stripped.startswith(b"{"):
            try:
                raw = json.loads(msg.data.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("invalid UTF-8 JSON audio envelope") from exc

            meta = raw.get("meta", raw)
            payload_b64 = raw.get("payload_b64") or raw.get("payload")

            if isinstance(payload_b64, str):
                payload = base64.b64decode(payload_b64)
            elif isinstance(payload_b64, list):
                payload = bytes(payload_b64)
            else:
                raise ValueError("JSON audio envelope missing payload_b64/payload")

            stream_name = meta.get("stream_name") or meta.get("stream") or ""
            if not stream_name:
                stream_name = (
                    msg.subject.rsplit(".", 1)[-1] if "." in msg.subject else "default"
                )

            capture_header = meta.get("source_capture_unix_ms")
            source_capture_unix_ms = (
                int(capture_header) if capture_header not in (None, "") else None
            )

            return cls(
                stream_name=stream_name,
                seq=int(meta.get("seq", 0)),
                timestamp_ms=int(meta.get("timestamp_ms", 0)),
                pts=int(meta.get("pts", 0)),
                codec=str(meta.get("codec", "pcm_s16le")),
                sample_rate=int(meta.get("sample_rate", 16000)),
                channels=int(meta.get("channels", 1)),
                payload=payload,
                source_capture_unix_ms=source_capture_unix_ms,
            )

        raise ValueError(
            f"unsupported audio envelope format: {len(msg.data)} bytes, subject={msg.subject!r}"
        )


@dataclass(slots=True)
class SubtitlePayload:
    kind: str
    stream_name: str
    cue_id: str
    text: str
    is_final: bool
    start: float = 0.0
    end: float = 0.0
    emission_time: float = 0.0
    source_seq: Optional[int] = None
    source_pts: Optional[int] = None
    source_timestamp_ms: Optional[int] = None
    source_capture_unix_ms: Optional[int] = None

    def to_wire_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")


def make_cue_id(stream_name: str, text: str, start: float, end: float) -> str:
    raw = f"{stream_name}|{round(start, 3)}|{round(end, 3)}|{text.strip()}".encode(
        "utf-8"
    )
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


def decode_audio_envelope(msg: Msg) -> AudioChunkEnvelope:
    return AudioChunkEnvelope.from_msg(msg)


def get_headers(msg: Msg) -> dict:
    headers = getattr(msg, "headers", None)
    if headers:
        return dict(headers)
    header = getattr(msg, "header", None)
    if header:
        return dict(header)
    return {}
