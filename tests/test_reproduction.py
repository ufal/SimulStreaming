"""Tests for reproduction setup validation."""
import pytest
import os


@pytest.mark.slow
def test_whisper_model_path_configurable():
    """Test that Whisper model path can be configured."""
    from simul_whisper.config import AlignAttConfig

    cfg = AlignAttConfig(
        model_path="/custom/path/to/model.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en"
    )

    assert cfg.model_path == "/custom/path/to/model.pt"


def test_supported_languages():
    """Test that system supports IWSLT language pairs."""
    from simul_whisper.config import AlignAttConfig

    # Test languages from IWSLT 2025
    languages = ["en", "de", "zh", "ja", "cs", "ar", "nl", "fr", "fa", "pt", "ru", "tr"]

    for lang in languages:
        cfg = AlignAttConfig(
            model_path="test.pt",
            segment_length=1.0,
            frame_threshold=25,
            language=lang
        )
        assert cfg.language == lang


def test_transcribe_and_translate_tasks():
    """Test that both transcribe and translate tasks are supported."""
    from simul_whisper.config import AlignAttConfig

    # Transcribe task
    cfg_transcribe = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        task="transcribe"
    )
    assert cfg_transcribe.task == "transcribe"

    # Translate task
    cfg_translate = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        task="translate"
    )
    assert cfg_translate.task == "translate"


def test_frame_threshold_configurations():
    """Test frame threshold values used in paper."""
    from simul_whisper.config import AlignAttConfig

    # Test different frame thresholds from paper
    thresholds = [15, 20, 25]

    for threshold in thresholds:
        cfg = AlignAttConfig(
            model_path="test.pt",
            segment_length=1.0,
            frame_threshold=threshold,
            language="en"
        )
        assert cfg.frame_threshold == threshold


def test_beam_search_configurations():
    """Test beam search configurations from paper."""
    from simul_whisper.config import AlignAttConfig

    # Greedy decoding (beam_size=1)
    cfg_greedy = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        beam_size=1,
        decoder_type="greedy"
    )
    assert cfg_greedy.beam_size == 1
    assert cfg_greedy.decoder_type == "greedy"

    # Beam search
    cfg_beam = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        beam_size=3,
        decoder_type="beam"
    )
    assert cfg_beam.beam_size == 3
    assert cfg_beam.decoder_type == "beam"


def test_context_and_prompt_settings():
    """Test context and prompt configurations."""
    from simul_whisper.config import AlignAttConfig

    cfg = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        init_prompt="Test prompt",
        static_init_prompt="Static terminology",
        max_context_tokens=100
    )

    assert cfg.init_prompt == "Test prompt"
    assert cfg.static_init_prompt == "Static terminology"
    assert cfg.max_context_tokens == 100


def test_audio_buffer_settings():
    """Test audio buffer configurations."""
    from simul_whisper.config import AlignAttConfig

    cfg = AlignAttConfig(
        model_path="test.pt",
        segment_length=1.0,
        frame_threshold=25,
        language="en",
        audio_max_len=30.0,
        audio_min_len=0.5
    )

    assert cfg.audio_max_len == 30.0
    assert cfg.audio_min_len == 0.5


@pytest.mark.slow
@pytest.mark.requires_model
def test_download_instructions():
    """Verify that model download instructions are accessible."""
    # This is a documentation test - just check that paths exist
    import simulstreaming_whisper

    # The system should be able to specify a model path
    # and download will happen automatically on first run
    assert hasattr(simulstreaming_whisper, 'simulwhisper_args')
