# SimulStreaming

SimulStreaming implements Whisper model for translation and transcription in
simultaneous mode (which is known as *streaming* in the ASR community).
SimulStreaming uses the state-of-the-art simultaneous policy AlignAtt, which
makes it very fast and efficient.

SimulStreaming merges [Simul-Whisper](https://github.com/backspacetg/simul_whisper/) and [Whisper-Streaming](https://github.com/ufal/whisper_streaming) projects.
Simul-Whisper implemented AlignAtt with Whisper, but only using large-v2 model
for transcription. We extend it with support for translation and large-v3 model, and with beam search, prompt for injecting in-domain
terminology, and context across the 30-second processing windows. Moreover,
Simul-Whisper implements only less realistic simulation on sentence-segmented
speech. Therefore, we use the interface of Whisper-Streaming for the long-form input
simulation, both computationally unaware and aware, and from both audio file and
simple demo TCP server that can be connected to microphone.

Moreover, SimulStreaming adds a machine translation model EuroLLM in a cascade, with LocalAgreement simultaneous policy, system
prompt, and in-context example.

SimulStreaming originates as Charles University (CUNI) submission to the IWSLT
2025 Simultaneous Shared Task. The results show that this system is extremely robust
and high quality. It is among the top performing systems in IWSLT 2025
Simultaneous Shared Task.

## Installation

The direct speech-to-text Whisper part can be installed with

```
pip install -r requirements.txt
```

The comments in `requirements.txt` document the origin of dependencies. There is originally WhisperStreaming code inserted in the `whisper_streaming` dir. It is simplified and refactored.
Simul-Whisper code is in `simul_whisper`, it includes the [original Whisper](https://github.com/openai/whisper) code adapted for SimulWhisper in `simul_whispre/whisper`.

**Lighter installation**

For slightly lighter installation,  remove `torchaudio` from `requirements.txt`. Then you can not use the Silero VAD controller (`--vac` option).

**Text-to-Text Translation**

Follow [translate/README.txt](translate/README.txt).

## Usage 

### Real-time simulation from audio file




