from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def decode_audio_file(path: str | Path, target_sample_rate: int = 16000) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed for {path}: {err}")

    if not proc.stdout:
        return np.zeros(0, dtype=np.float32)

    audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
    return np.ascontiguousarray(audio, dtype=np.float32)
