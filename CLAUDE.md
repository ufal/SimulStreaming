# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Fork Information

This is a fork of **ufal/SimulStreaming** (upstream).

Git remotes:
- `origin`: git@github.com:krystophny/SimulStreaming.git
- `upstream`: git@github.com:ufal/SimulStreaming.git

To sync with upstream:
```bash
git fetch upstream
git merge upstream/main
```

## Project Overview

SimulStreaming implements Whisper model for simultaneous speech translation and transcription using the AlignAtt simultaneous policy. It merges Simul-Whisper and Whisper-Streaming projects, extending them with support for translation, large-v3 model, beam search, prompt injection, and context across 30-second windows.

The system can optionally cascade with EuroLLM machine translation using LocalAgreement policy (see `translate/` directory).

## Installation and Dependencies

### Development Installation

For development with testing:
```bash
pip install -e ".[dev]"
```

For regular installation:
```bash
pip install -e .
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

Dependencies:
- **torch**: Core ML framework (from SimulWhisper)
- **librosa**: Audio processing (WhisperStreaming)
- **torchaudio**: Required for Silero VAD (`--vac` option). Can be removed for lighter installation if VAD not needed.
- **tqdm, tiktoken**: From Whisper
- **triton>=2.0.0**: Required by both Whisper and SimulWhisper

Development dependencies (included in `[dev]`):
- **pytest>=7.0**: Testing framework
- **pytest-cov>=4.0**: Coverage reporting
- **pytest-timeout>=2.1**: Test timeouts

## Usage Commands

### Real-time Simulation from Audio File

```bash
# Basic transcription
python3 simulstreaming_whisper.py audio.wav --language cs --task translate --comp_unaware

# With VAC (voice activity controller) - recommended
python3 simulstreaming_whisper.py audio.wav --language en --task transcribe --vac

# Start at specific time (for debugging)
python3 simulstreaming_whisper.py audio.wav --language de --task translate --start_at 120.5
```

**Simulation modes:**
- Default: Real-time computationally aware (chunk size = `MIN_CHUNK_SIZE` or larger if processing delayed)
- `--comp_unaware`: Computationally unaware (timer "stops" during computation, shows lower-bound latency)
- `--start_at START_AT`: Start processing at specific time (useful for debugging specific audio segments)

### Server Mode (Real-time from Microphone)

Start server:
```bash
python3 simulstreaming_whisper_server.py --host localhost --port 43001 --language en --task transcribe --vac --warmup-file warmup.wav
```

Connect client (Linux):
```bash
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

**Note:** `--warmup-file` is important - first chunk processing can be slow without warmup.

### Translation Cascade with EuroLLM

See `translate/README.txt` for setup. Requires CTranslate2 and EuroLLM model conversion.

## Architecture

### Core Components

**Entry points:**
- `simulstreaming_whisper.py`: File simulation mode
- `simulstreaming_whisper_server.py`: TCP server for live audio

**Main implementation (`simul_whisper/`):**
- `simul_whisper.py`: Core `PaddedAlignAttWhisper` class implementing AlignAtt policy with encoder-decoder attention hooks, KV caching, and beam search
- `config.py`: `AlignAttConfig` dataclass with all model configuration
- `beam.py`: Beam search implementation (`BeamPyTorchInference`)
- `eow_detection.py`: End-of-word detection using CIF model (truncates incomplete words)
- `generation_progress.py`: Tracks generation progress with attention frames
- `whisper/`: Modified OpenAI Whisper code for SimulWhisper compatibility

**Streaming interface (`whisper_streaming/`):**
- `base.py`: `ASRBase` and `OnlineProcessorInterface` abstract base classes
- `whisper_online_main.py`: Main simulation loop
- `vac_online_processor.py`: Voice activity controller wrapper
- `silero_vad_iterator.py`: Silero VAD implementation

**Integration:**
- `simulstreaming_whisper.py` defines `SimulWhisperASR` (extends `ASRBase`) and `SimulWhisperOnline` (extends `OnlineProcessorInterface`)
- `SimulWhisperOnline.process_iter()` is the core iteration method - inserts audio, runs inference, handles unicode, returns timestamped text
- AlignAtt policy in `simul_whisper.py` uses encoder-decoder attention to determine when to stop decoding (only decode until `frame_threshold` frames from audio end)

### Key Architectural Patterns

1. **Streaming Processing**: Audio processed in chunks with stateful buffer management. Audio accumulates in `audio_chunks`, processed via `process_iter()`, state maintained across iterations.

2. **AlignAtt Simultaneous Policy**: Decoder uses encoder-decoder cross-attention to determine when to stop (decodes only until `frame_threshold` frames from end). Attention hooks capture alignment between decoded tokens and audio frames.

3. **End-of-Word Detection**: CIF model (`--cif_ckpt_path`) detects word boundaries. Incomplete words at chunk end are truncated to avoid partial word errors. Controlled via `--never_fire` flag.

4. **Context Management**: Token buffer (`token_buffer.py`) maintains context across 30-second windows for coherence. Supports static prompt (`--static_init_prompt`) and scrolling context (`--init_prompt`, `--max_context_tokens`).

5. **Unicode Handling**: `hide_incomplete_unicode()` in `SimulWhisperOnline` prevents incomplete UTF-8 characters at chunk boundaries (stores in `unicode_buffer` for next iteration).

## Important Configuration Parameters

- `--frame_threshold`: AlignAtt policy threshold (frames from audio end where decoding stops). One frame = 0.02s for large-v3. Lower = more eager decoding = lower latency but potentially lower quality.
- `--audio_max_len`: Max audio buffer length (default 30.0s, Whisper's processing window)
- `--audio_min_len`: Skip processing if buffer shorter than this (useful with small `--min-chunk-size`)
- `--min-chunk-size`: Minimum audio chunk size for processing updates
- `--beams`: Beam search width (1 = greedy decoding)
- `--cif_ckpt_path`: CIF model for end-of-word detection. Models available at https://github.com/backspacetg/simul_whisper/tree/main/cif_models (no large-v3 model available)
- `--never_fire`: Override CIF model to never truncate last word

## Output Format

Space-separated columns (file simulation):
1. Emission time (ms) - [omitted in server mode]
2. Start timestamp (ms) in original audio
3. End timestamp (ms) in original audio
4. Text (starts with space if appended, or punctuation to concatenate)

Example:
```
1200.0000 0 1200  And so
2400.0000 1200 2400  my fellow Americans
3600.0000 2400 3600 ,
```

## Model Files

- `--model_path`: Whisper .pt model file (default `./large-v3.pt`). Auto-downloaded if not present.
- Models: large-v2, large-v3 supported (translation and transcription)
- CIF models: Separate per Whisper version, none available for large-v3

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=simul_whisper --cov=whisper_streaming --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_imports.py
```

Run tests excluding slow tests:
```bash
pytest -m "not slow"
```

Run tests excluding those requiring models:
```bash
pytest -m "not requires_model"
```

### Test Structure

- `tests/test_imports.py`: Module import tests
- `tests/test_config.py`: Configuration class tests
- `tests/test_triton_fallback.py`: Triton ops fallback behavior
- `tests/test_argument_parsing.py`: CLI argument parsing
- `tests/test_base_classes.py`: Base class interface tests
- `tests/test_token_buffer.py`: Token buffer functionality

### Test Markers

- `@pytest.mark.slow`: Slow tests (can be skipped)
- `@pytest.mark.requires_model`: Tests requiring model download
- `@pytest.mark.requires_cuda`: Tests requiring CUDA

## Reproducing IWSLT 2025 Results

See **[REPRODUCTION.md](REPRODUCTION.md)** for detailed instructions on reproducing the CUNI IWSLT 2025 results, including:
- Downloading the ACL 60-60 evaluation dataset
- Setting up models (Whisper large-v3, EuroLLM)
- Running evaluations with proper configurations
- Computing BLEU scores and latency metrics
- Expected performance benchmarks

## Development Notes

- Code origin: `simul_whisper/whisper/` is modified OpenAI Whisper, `whisper_streaming/` is refactored WhisperStreaming
- Logging: Use `--log-level DEBUG` for detailed debugging, especially useful with `--start_at` for specific audio segments
- Testing VAC: VAC (`--vac`) requires torchaudio and improves quality by handling silence/speech boundaries
- Translation cascade: Optional EuroLLM integration in `translate/` for speech-to-text-to-translation pipeline
- **Rebasing**: When rebasing on upstream, preserve the following new files: `pyproject.toml`, `tests/`, `.gitignore`, `whisper_streaming/__init__.py`, `REPRODUCTION.md`. Do not move or restructure existing upstream files.
