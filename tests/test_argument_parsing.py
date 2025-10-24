"""Test command-line argument parsing."""
import pytest
import argparse
from simulstreaming_whisper import simulwhisper_args


def test_argument_parsing_defaults():
    """Test argument parsing with defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path')
    parser.add_argument('--min-chunk-size', type=float, default=1.0)
    parser.add_argument('--lan', '--language', type=str, default='en')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='transcribe')
    parser.add_argument('--vac', action='store_true')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04)
    parser.add_argument('-l', '--log-level', type=str, default='INFO')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--start_at', type=float, default=0)
    parser.add_argument('--comp_unaware', action='store_true')

    simulwhisper_args(parser)

    args = parser.parse_args(['test.wav'])

    assert args.audio_path == 'test.wav'
    assert args.model_path == './large-v3.pt'
    assert args.beams == 1
    assert args.frame_threshold == 25
    assert args.audio_max_len == 30.0
    assert args.audio_min_len == 0.0


def test_argument_parsing_custom():
    """Test argument parsing with custom values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path')
    parser.add_argument('--min-chunk-size', type=float, default=1.0)
    parser.add_argument('--lan', '--language', type=str, default='en')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='transcribe')
    parser.add_argument('--vac', action='store_true')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04)
    parser.add_argument('-l', '--log-level', type=str, default='INFO')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--start_at', type=float, default=0)
    parser.add_argument('--comp_unaware', action='store_true')

    simulwhisper_args(parser)

    args = parser.parse_args([
        'test.wav',
        '--language', 'de',
        '--task', 'translate',
        '--beams', '3',
        '--frame_threshold', '15',
        '--audio_max_len', '25.0'
    ])

    assert args.audio_path == 'test.wav'
    assert args.lan == 'de'
    assert args.task == 'translate'
    assert args.beams == 3
    assert args.frame_threshold == 15
    assert args.audio_max_len == 25.0


def test_argument_parsing_prompts():
    """Test argument parsing with prompts."""
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path')
    parser.add_argument('--min-chunk-size', type=float, default=1.0)
    parser.add_argument('--lan', '--language', type=str, default='en')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='transcribe')
    parser.add_argument('--vac', action='store_true')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04)
    parser.add_argument('-l', '--log-level', type=str, default='INFO')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--start_at', type=float, default=0)
    parser.add_argument('--comp_unaware', action='store_true')

    simulwhisper_args(parser)

    args = parser.parse_args([
        'test.wav',
        '--init_prompt', 'Hello world',
        '--static_init_prompt', 'terminology: AI, ML',
        '--max_context_tokens', '100'
    ])

    assert args.init_prompt == 'Hello world'
    assert args.static_init_prompt == 'terminology: AI, ML'
    assert args.max_context_tokens == 100
