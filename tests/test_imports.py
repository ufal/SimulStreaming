"""Test that all modules can be imported."""
import pytest


def test_import_simul_whisper():
    """Test simul_whisper module imports."""
    import simul_whisper
    assert simul_whisper is not None


def test_import_config():
    """Test AlignAttConfig can be imported."""
    from simul_whisper.config import AlignAttConfig
    assert AlignAttConfig is not None


def test_import_simul_whisper_class():
    """Test PaddedAlignAttWhisper can be imported."""
    from simul_whisper.simul_whisper import PaddedAlignAttWhisper
    assert PaddedAlignAttWhisper is not None


def test_import_whisper_streaming():
    """Test whisper_streaming module imports."""
    import whisper_streaming.base
    assert whisper_streaming.base is not None


def test_import_asr_base():
    """Test ASRBase and OnlineProcessorInterface can be imported."""
    from whisper_streaming.base import ASRBase, OnlineProcessorInterface
    assert ASRBase is not None
    assert OnlineProcessorInterface is not None


def test_import_simulstreaming_whisper():
    """Test main entry point imports."""
    from simulstreaming_whisper import SimulWhisperASR, SimulWhisperOnline
    assert SimulWhisperASR is not None
    assert SimulWhisperOnline is not None


def test_import_triton_ops():
    """Test triton_ops can be imported (may fail on non-CUDA systems)."""
    try:
        from simul_whisper.whisper import triton_ops
        assert triton_ops is not None
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Triton import failed (expected on non-CUDA): {e}")


def test_import_beam():
    """Test beam search can be imported."""
    from simul_whisper.beam import BeamPyTorchInference
    assert BeamPyTorchInference is not None


def test_import_eow_detection():
    """Test end-of-word detection can be imported."""
    from simul_whisper.eow_detection import fire_at_boundary, load_cif
    assert fire_at_boundary is not None
    assert load_cif is not None


def test_import_generation_progress():
    """Test generation progress tracking can be imported."""
    from simul_whisper import generation_progress
    assert generation_progress is not None
