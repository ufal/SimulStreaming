"""Test base classes for ASR and online processing."""
import pytest
from whisper_streaming.base import ASRBase, OnlineProcessorInterface


def test_online_processor_interface_constants():
    """Test OnlineProcessorInterface has expected constants."""
    assert hasattr(OnlineProcessorInterface, 'SAMPLING_RATE')
    assert OnlineProcessorInterface.SAMPLING_RATE == 16000


def test_online_processor_interface_methods():
    """Test OnlineProcessorInterface has expected methods."""
    assert hasattr(OnlineProcessorInterface, 'insert_audio_chunk')
    assert hasattr(OnlineProcessorInterface, 'process_iter')
    assert hasattr(OnlineProcessorInterface, 'finish')


def test_asr_base_separator():
    """Test ASRBase has separator attribute."""
    assert hasattr(ASRBase, 'sep')
    assert ASRBase.sep == " "


def test_asr_base_methods():
    """Test ASRBase has expected methods."""
    assert hasattr(ASRBase, 'transcribe')
    assert hasattr(ASRBase, 'warmup')
    assert hasattr(ASRBase, 'use_vad')
    assert hasattr(ASRBase, 'set_translate_task')
