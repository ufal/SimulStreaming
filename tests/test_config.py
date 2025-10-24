"""Test configuration classes."""
import pytest
from simul_whisper.config import AlignAttConfig


def test_alignatt_config_defaults():
    """Test AlignAttConfig can be created with defaults."""
    cfg = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en"
    )

    assert cfg.model_path == "test.pt"
    assert cfg.segment_length == 1.0
    assert cfg.frame_threshold == 25
    assert cfg.language == "en"
    assert cfg.audio_max_len == 30.0  # default
    assert cfg.audio_min_len == 1.0  # default


def test_alignatt_config_custom():
    """Test AlignAttConfig with custom parameters."""
    cfg = AlignAttConfig(
        model_path="large-v3.pt",
        segment_length=2.0,
        frame_threshold=15,
        language="de",
        audio_max_len=25.0,
        audio_min_len=0.5,
        beam_size=3,
        task="translate"
    )

    assert cfg.model_path == "large-v3.pt"
    assert cfg.segment_length == 2.0
    assert cfg.frame_threshold == 15
    assert cfg.language == "de"
    assert cfg.audio_max_len == 25.0
    assert cfg.audio_min_len == 0.5
    assert cfg.beam_size == 3
    assert cfg.task == "translate"


def test_alignatt_config_auto_language():
    """Test AlignAttConfig with automatic language detection."""
    cfg = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="auto"
    )

    assert cfg.language == "auto"
