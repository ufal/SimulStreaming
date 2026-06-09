from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Stream Registry", version="1.0.0")

_lock = Lock()
_streams: dict[str, dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StreamRegisterRequest(BaseModel):
    path: str
    display_name: str
    source_type: str = "ffmpeg"
    source_id: Optional[str] = None
    rtsp_url: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StreamUnregisterRequest(BaseModel):
    path: str
    source_id: Optional[str] = None


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "createdAt": item.get("created_at"),
        "updatedAt": item.get("updated_at"),
        "displayName": item.get("display_name"),
        "sourceType": item.get("source_type"),
        "sourceId": item.get("source_id"),
        "rtspUrl": item.get("rtsp_url"),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/streams")
def list_streams():
    with _lock:
        items = sorted(_streams.values(), key=lambda x: x["path"])
        return {"items": [normalize_item(item) for item in items]}


@app.post("/stream/register")
def register_stream(payload: StreamRegisterRequest):
    with _lock:
        ts = now_iso()
        item = _streams.get(payload.path)

        if item is None:
            _streams[payload.path] = {
                "path": payload.path,
                "display_name": payload.display_name,
                "source_type": payload.source_type,
                "source_id": payload.source_id,
                "rtsp_url": payload.rtsp_url,
                "created_at": ts,
                "updated_at": ts,
                "status": "active",
                "extra": payload.extra,
            }
        else:
            item.update(
                {
                    "display_name": payload.display_name,
                    "source_type": payload.source_type,
                    "source_id": payload.source_id,
                    "rtsp_url": payload.rtsp_url,
                    "updated_at": ts,
                    "status": "active",
                    "extra": payload.extra,
                }
            )

        return {"ok": True, "item": normalize_item(_streams[payload.path])}


@app.post("/stream/unregister")
def unregister_stream(payload: StreamUnregisterRequest):
    with _lock:
        item = _streams.get(payload.path)
        if item is None:
            return {"ok": True, "removed": False}

        if payload.source_id is not None and item.get("source_id") not in (
            None,
            payload.source_id,
        ):
            raise HTTPException(status_code=409, detail="source_id mismatch")

        del _streams[payload.path]
        return {"ok": True, "removed": True}
