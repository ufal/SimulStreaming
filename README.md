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


```
usage: simulstreaming_whisper.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--lan LAN] [--task {transcribe,translate}] [--vac] [--vac-chunk-size VAC_CHUNK_SIZE] [--vad] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--model_path MODEL_PATH] [--beams BEAMS]
                                 [--decoder DECODER] [--audio_max_len AUDIO_MAX_LEN] [--audio_min_len AUDIO_MIN_LEN] [--frame_threshold FRAME_THRESHOLD] [--cif_ckpt_path CIF_CKPT_PATH] [--never_fire | --no-never_fire] [--init_prompt INIT_PROMPT]
                                 [--static_init_prompt STATIC_INIT_PROMPT] [--max_context_tokens MAX_CONTEXT_TOKENS] [--start_at START_AT] [--comp_unaware]
                                 audio_path

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the log level

WhisperStreaming processor arguments (shared for simulation from file and for the server):
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
  --lan LAN, --language LAN
                        Source language code, e.g. en,de,cs.
  --task {transcribe,translate}
                        Transcribe or translate.
  --vac                 Use VAC = voice activity controller. Recommended. Requires torch.
  --vac-chunk-size VAC_CHUNK_SIZE
                        VAC sample size in seconds.
  --vad                 Use VAD = voice activity detection, with the default parameters.

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
                        Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model.

Truncation of the last decoded word (from Simul-Whisper):
  --cif_ckpt_path CIF_CKPT_PATH
                        The file path to the Simul-Whisper's CIF model checkpoint that detects whether there isend of word at the end of the chunk. If not, the last decoded space-separated word is truncated because it is often wrong -- transcribing a
                        word in the middle.The CIF model adapted for the Whisper model version should be used. Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . Note that there is no model for large-v3.
  --never_fire, --no-never_fire
                        Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. . If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. Otherwise, if
                        the CIF model path is not set, the last word is ALWAYS trimmed. (default: False)

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


**Usage as a server with mic input, or as a module:** TODO. Analogical to WhisperStreaming.

### Output format

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
- column 1: the emission time of that line, in miliseconds. In `--comp_unaware` mode, it's the simulated time.
- columns 2-3: the beginning and end timestamp of the line in original audio. (TODO: it should be, currently it is very rough approximation.)
- column 4: This column starts either with a space, if the previous line had to be appended with a space, or with a character that has to be appended to the previous line (like comma or dot).


## Feedback welcome!

We, the authors of SimulStreaming from Charles University,
wish to continue with our research and do it better. We see an opportunity to
involve feedback from those who use SimulStreaming in our future work.
Therefore, we kindly ask all users of SimulStreaming to fill a questionnaire at this link: TODO. We wish to learn from you:

- who are you? Student/teacher/academic/private person/non-profit/enterpreneur, founder/small or medium enterprise/enterprise that is not small or medium
- what is your intent in using SimulStreaming? On what kind of content, what language(s), translate or transcribe? What latency?
- what are the biggest challenges or issues you face? 
- what HW? laptop/GPU server/HPC cluster/mobile or other devices?
- do you wish more services or support of the author?
 - replying on GitHub issues
 - consultation whether SimulStreaming is the right choice for me
 - support with installation on my HW
 - advice for optimal model and parameters
 - experiments for optimal model and parameters
 - SimulStreaming via API (software as a service)
 - SimulStreaming in an application, including front-end

Contributions:
- would you contribute to SimulStreaming with bug fixes and new features?
 - yes. And I would permit authors to use my code contribution in any way
 - yes only if SimulStreaming would be licenced freely and openly, e.g. under MIT or Apache 2
 - no
 - other: ...

Commercialization:
- are you considering to use SimulStreaming commercially? yes/no

If yes:
- would you go acquire commercial licence for SimulStreaming through these steps?
 - Getting a free commercial licence from GitHub directly, without anything else
 - Register who you are, get a free unlimited commercial licence
 - Register, pay symbolic 1 euro, get an unlimited commercial licence
 - Register, pay an amount I choose, get an unlimited commercial licence
 - Register, pay, get an unlimited commercial licence and a time-limited support
 - Register, pay a subscription plan for SimulStreaming via API

- How much would you pay for commercial licence of SimulStreaming? 

- Do you wish to be notified as soon as the commercial licence distribution is ready? yes/no
- Do you wish to be contacted by the authors for a feedback? yes+who you are+contact / no

Moreover, to collect valuable statistics, publish this code openly under a non-commercial licence. 



## Licence and Contributions

This code is dual-licensed. You can use SimulStreaming under PolyForm Noncommercial License 1.0.0, if you acquire a copy of this code through GitHub repo. 

**Please, give feedback** We, the authors of SimulStreaming at Charles University,
wish to continue with our research and do it better. We see an opportunity to
involve feedback from the users of SimulStreaming in future work. 

Therefore, we kindly ask all users of SimulStreaming to fill a questionnaire at this link: 

**Commercial licence** We are considering to provide commercial licenses for a symbolic fee or for free.
If are interested to use SimulStreaming commercially,
*please fill the questionnaire and leave us your contact. We are considering to provide commercial licenses for a symbolic fee or for free.


Therefore, we plan to provide commercial licences to those who
register and voluntarily also share their feedback or pay a small symbolic fee.

**Commercial licence**
The authors are currently working on a
seamless authomatic way to distribute the commercial licences, and on improving their service to users.

The authors are planning easy automatic way to distribute the licences. 

Read more information in [COMMERCIAL.md](COMMERCIAL.md).

**Contributions** are welcome. You will be asked to 

## Contact

Dominik Macháček, machacek@ufal.mff.cuni.cz

