"""Pytest configuration and fixtures for SimulStreaming tests."""
import pytest
import numpy as np
import torch
import wave
import tempfile
import os


@pytest.fixture
def sample_audio_array():
    """Create a simple audio array for testing (3 seconds, 16kHz, 440Hz tone)."""
    sample_rate = 16000
    duration = 3
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_file(sample_audio_array, tmp_path):
    """Create a temporary WAV file for testing."""
    audio_path = tmp_path / "test_audio.wav"
    audio_int16 = (sample_audio_array * 32767).astype(np.int16)

    with wave.open(str(audio_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_int16.tobytes())

    return audio_path


@pytest.fixture
def sample_torch_audio(sample_audio_array):
    """Create a torch tensor from sample audio."""
    return torch.from_numpy(sample_audio_array)


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()
