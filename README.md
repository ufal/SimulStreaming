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

SimulStreaming originates as [Charles University (CUNI) submission to the IWSLT
2025 Simultaneous Shared Task](https://arxiv.org/abs/2506.17077). The results show that this system is extremely robust
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


```
usage: simulstreaming_whisper.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--lan LAN] [--task {transcribe,translate}] [--vac] [--vac-chunk-size VAC_CHUNK_SIZE]
                                 [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--model_path MODEL_PATH] [--beams BEAMS] [--decoder DECODER] [--audio_max_len AUDIO_MAX_LEN]
                                 [--audio_min_len AUDIO_MIN_LEN] [--frame_threshold FRAME_THRESHOLD] [--cif_ckpt_path CIF_CKPT_PATH] [--never_fire | --no-never_fire]
                                 [--init_prompt INIT_PROMPT] [--static_init_prompt STATIC_INIT_PROMPT] [--max_context_tokens MAX_CONTEXT_TOKENS] [--start_at START_AT] [--comp_unaware]
                                 audio_path

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the log level

WhisperStreaming processor arguments (shared for simulation from file and for the server):
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole
                        segment that was received by this time.
  --lan LAN, --language LAN
                        Source language code, e.g. en, de, cs, or auto for automatic language detection from speech.
  --task {transcribe,translate}
                        Transcribe or translate.
  --vac                 Use VAC = voice activity controller. Recommended. Requires torch.
  --vac-chunk-size VAC_CHUNK_SIZE
                        VAC sample size in seconds.

Whisper arguments:
  --model_path MODEL_PATH
                        The file path to the Whisper .pt model. If not present on the filesystem, the model is downloaded automatically.
  --beams BEAMS, -b BEAMS
                        Number of beams for beam search decoding. If 1, GreedyDecoder is used.
  --decoder DECODER     Override automatic selection of beam or greedy decoder. If beams > 1 and greedy: invalid.

Audio buffer:
  --audio_max_len AUDIO_MAX_LEN
                        Max length of the audio buffer, in seconds.
  --audio_min_len AUDIO_MIN_LEN
                        Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.

AlignAtt argument:
  --frame_threshold FRAME_THRESHOLD
                        Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02
                        seconds for large-v3 model.

Truncation of the last decoded word (from Simul-Whisper):
  --cif_ckpt_path CIF_CKPT_PATH
                        The file path to the Simul-Whisper's CIF model checkpoint that detects whether there isend of word at the end of the chunk. If not, the last decoded space-
                        separated word is truncated because it is often wrong -- transcribing a word in the middle.The CIF model adapted for the Whisper model version should be used. Find
                        the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . Note that there is no model for large-v3.
  --never_fire, --no-never_fire
                        Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. . If False: if CIF model path is set, the last word is
                        SOMETIMES truncated, depending on the CIF detection. Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed. (default: False)

Prompt and context:
  --init_prompt INIT_PROMPT
                        Init prompt for the model. It should be in the target language.
  --static_init_prompt STATIC_INIT_PROMPT
                        Do not scroll over this text. It can contain terminology that should be relevant over all document.
  --max_context_tokens MAX_CONTEXT_TOKENS
                        Max context tokens for the model. Default is 0.

Arguments for simulation from file:
  audio_path            Filename of 16kHz mono channel wav, on which live streaming is simulated.
  --start_at START_AT   Start processing audio at this time.
  --comp_unaware        Computationally unaware simulation.
```

Example:

```
python3 simulstreaming_whisper.py audio.wav --language cs  --task translate --comp_unaware
```

Simulation modes:

- default mode, no special option: real-time simulation from file, computationally aware. The chunk size is `MIN_CHUNK_SIZE` or larger, if more audio arrived during last update computation.

- `--comp_unaware` option: computationally unaware simulation. It means that the timer that counts the emission times "stops" when the model is computing. The chunk size is always `MIN_CHUNK_SIZE`. The latency is caused only by the model being unable to confirm the output, e.g. because of language ambiguity etc., and not because of slow hardware or suboptimal implementation. We implement this feature for finding the lower bound for latency.

- `--start_at START_AT`: Start processing audio at this time. The first update receives the whole audio by `START_AT`. It is useful for debugging, e.g. when we observe a bug in a specific time in audio file, and want to reproduce it quickly, without long waiting.

- offline mode, to process whole audio with maximum quality, is not available yet. Instead, try large `--min-chunk-size` and `--frame-threshold`.


### Server -- real-time from mic 

The entry point `simulstreaming_whisper_server.py` has the same model options as `simulstreaming_whisper.py`, plus `--host` and `--port` of the TCP connection and the `--warmup-file`. The warmup file is decoded by the Whisper backend after the model is loaded because without that, processing of the very the first input chunk may take longer.

See the help message (`-h` option).

**Linux** client example:

```
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

- `arecord` sends realtime audio from a sound device (e.g. mic), in raw audio format -- 16000 sampling rate, mono channel, S16_LE -- signed 16-bit integer low endian. (Or other operating systems, use another alternative)

- nc is netcat with server's host and port

**Windows/Mac**: `ffmpeg` may substitute `arecord`. Or use the solutions proposed in Whisper-Streaming pull requests [#111](https://github.com/ufal/whisper_streaming/pull/111) and [#123](https://github.com/ufal/whisper_streaming/pull/123).



### Output format

This is example of the output format of the simulation from file. The output from the server is the same except that the first space-separated column is not there.

```
1200.0000 0 1200  And so
2400.0000 1200 2400  my fellow Americans
3600.0000 2400 3600 ,
4800.0000 3600 4800  ask not
6000.0000 4800 6000  what
7200.0000 6000 7200  your country can do
8400.0000 7200 8400  for you,
9600.0000 8400 9600  ask what you
10800.0000 9600 10800  can do for your country
11000.0000 10800 11000 .
```

It's space-separated. The first three columns are:
- column 1: the emission time of that line, in miliseconds. In `--comp_unaware` mode, it's the simulated time. In server, this column is not there.
- columns 2-3: the beginning and end timestamp of the line in original audio. (TODO: it should be, currently it is very rough approximation.)
- columns 4-: This column starts either with a space, if the previous line had to be appended with a space, or with a character that has to be appended to the previous line (like comma or dot).



## 📣 Feedback Welcome!

We, the authors of SimulStreaming from Charles University, are committed to
improving our research and the tool itself. Your experience as a user is
invaluable to us --- it can help to shape upcoming features, licensing models, and support services. 

To better understand your needs and guide the future of
SimulStreaming, we kindly ask the users, especially commercial, to fill out this **[questionnaire](https://forms.cloud.microsoft/e/7tCxb4gJfB).**

## 📄 Licence

Now under MIT.

## 🤝 Contributions

Contributions to SimulStreaming are welcome. 

## ✉️ Contact

[Dominik Macháček](https://ufal.mff.cuni.cz/dominik-machacek/), machacek@ufal.mff.cuni.cz

