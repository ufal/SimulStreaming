# SimulStreaming

TODO: what is it

## Installation

This code extends WhisperStreaming and SimulStreaming which extends Whisper. 

`pip install -r requirements.txt`

The dependent libraries are cies are WhisperSimulWhisper and Whisper -- both

### Light installation

Lighter installation can avoid `torchaudio` by removing it from `requirements.txt`. Then the Silero VAD controller (`--vac` option) can not be used.