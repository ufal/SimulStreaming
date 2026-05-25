# SimulStreaming Canary with Alignatt

SimulStreaming is a tool for simultaneous (aka streaming) processing of speech-to-text and LLM translation models. 

## How to use SimulStreaming Canary

### 1. Canary speech-to-text

#### Installation

```bash
pip install -r requirements_canary.txt
```
**Lighter installation**

For slightly lighter installation, remove `torchaudio`. Then you can not use the Silero VAD controller (`--vac` option).

#### Usage: Real-time simulation from audio file

```
$ python3 simulstreaming_canary.py -h
usage: simulstreaming_canary.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--source_lang LAN] [--target_lang LAN] [--vac] [--vac-chunk-size VAC_CHUNK_SIZE]
                                 [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--logdir LOGDIR] [--out-txt] [--model_path MODEL_PATH] [--beams BEAMS] [--decoder DECODER]
                                 [--audio_max_len AUDIO_MAX_LEN] [--audio_min_len AUDIO_MIN_LEN] [--frame_threshold FRAME_THRESHOLD]
                                 [--start_at START_AT] [--comp_unaware]
                                 audio_path

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the log level
  --logdir LOGDIR       Directory to save audio segments and generated texts for debugging.
  --out-txt             Output formatted as not as jsonl but simple space-separated text: beg, end, text

WhisperStreaming processor arguments (shared for simulation from file and for the server):
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole
                        segment that was received by this time.
                        Source language code, e.g. en, de, cs, or auto for automatic language detection from speech.
  --vac                 Use VAC = voice activity controller. Recommended. Requires torch.
  --vac-chunk-size VAC_CHUNK_SIZE
                        VAC sample size in seconds.

Canary arguments:
  --model_path MODEL_PATH
                        The file path to the Whisper .nemo model. If not present on the filesystem, the model is downloaded automatically.
  --beams BEAMS, -b BEAMS
                        Number of beams for beam search decoding. If 1, GreedyDecoder is used.
  --decoder DECODER     Override automatic selection of beam or greedy decoder. If beams > 1 and greedy: invalid.
  --source_lang LAN
                        Source language.
  --target_lang LAN
                        Target language. Same as source for transcription.

Audio buffer:
  --audio_max_len AUDIO_MAX_LEN
                        Max length of the audio buffer, in seconds.
  --audio_min_len AUDIO_MIN_LEN
                        Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.

AlignAtt argument:
  --frame_threshold FRAME_THRESHOLD
                        Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02
                        seconds for large-v3 model.
  --strip_incomplete_words
                        If set, trailing incomplete words are stripped
                        from the AlignAtt output before emission.

Prompt and context:
  --max_context_tokens MAX_CONTEXT_TOKENS
                        Max context tokens for the model. Default is 0.
  --decoder_context
                        A flag whether to use decoder context. Experiments showed it to be problematic.

Arguments for simulation from file:
  audio_path            Filename of 16kHz mono channel wav, on which live streaming is simulated.
  --start_at START_AT   Start processing audio at this time.
  --comp_unaware        Computationally unaware simulation.
```

**Example:**

```
python3 simulstreaming_canary.py audio.wav --source_lang en --target_lang en --comp_unaware --vac
```

**Simulation modes:**

- default mode, no special option: real-time simulation from file, computationally aware. The chunk size is `MIN_CHUNK_SIZE` or larger, if more audio arrived during last update computation.

- `--comp_unaware` option: computationally unaware simulation. It means that the timer that counts the emission times "stops" when the model is computing. The chunk size is always `MIN_CHUNK_SIZE`. The latency is caused only by the model being unable to confirm the output, e.g. because of language ambiguity etc., and not because of slow hardware or suboptimal implementation. We implement this feature for finding the lower bound for latency.

- `--start_at START_AT`: Start processing audio at this time. The first update receives the whole audio by `START_AT`. It is useful for debugging, e.g. when we observe a bug in a specific time in audio file, and want to reproduce it quickly, without long waiting.

- offline mode, to process whole audio with maximum quality, is not available yet. Instead, try large `--min-chunk-size` and `--frame-threshold`.


**Output format:**

1) Default JSONL

Default stdout is jsonl. The command incrementally processes incoming audio chunks. After each chunk, there can be either:
- no output
- a line with partial text output, like

```
{"start": 0.332, "end": 0.832, "text": " And so,", "tokens": [400, 370, 11], "words": [{"start": 0.332, "end": 0.332, "text": " And", "tokens": [400]}, {"start": 0.532, "end": 0.532, "text": " so", "tokens": [370]}, {"start": 0.832, "end": 0.832, "text": ",", "tokens": [11]}], "is_final": false, "emission_time": 2.246028423309326}
```

- a line indicating end of voiced segment without any text update: 

```
{"is_final": true, "emission_time": 3.060192108154297}
```

Explanation of json fields:

```
{
  # start and end timestamps, in seconds, of a segment of the source audio where this partial output was detected by Whisper model. 
  # Warning: it may be inaccurate but good enough for some applications.
  "start": 0,  
  "end": 0.68,  


  # the partial text output produced in this update
  "text": " And so",  # the partial text output produced by this update
  "tokens": [  # list of token ids of text, used by Whisper tokenizer
    400,
    370
  ],

  # text segmented for more detailed word-level view
  "words": [  
    {
      "start": 0,  
      "end": 0,
      "text": " And",
      "tokens": [  
        400
      ]
    },
    {
      "start": 0.68,
      "end": 0.68,
      "text": " so",
      "tokens": [
        370
      ]
    }
  ],


  # a flag indicating end of voice at the end of segment in the original audio
  # this is used only with --vac option (Voice Activity Controller)
  "is_final": false,

  # simulation time, in seconds, when this line of output was produced
  # - in computational aware simulation: real time from the beginning of process  
  # - in computatational unaware simulation: length of incoming audio
  "emission_time": 1.9224984645843506
}
```

2. Simple text, with `--out-txt` option:

For back dependency, and for better eye-readable output for debugging, there is option for this format:

```
2246.5429 332 832  And so,
3468.0274 1032 1712  my fellow Americans
4637.6612 2172 3272 , ask
```

On each line, there are 3 space-delimited columns: 
- 1. emission time, in milliseconds
- 2.-3. start and end timestamps, in milliseconds

After these three columns, there is one space, and then the text. Notice that the text may start with a space, as the first line " And so,", or may not, as the 3rd line ", ask".

End of voice or word-level segments are not indicated in this format.

**Debug: Logdir**

With `--logdir LOGDIR` and `--vac` parameters, the tool will create a directory named LOGDIR. In this dir, there will be subdirectories for each voiced segment. Inside, for each chunk update, there will be: 
- an audio file with exact content of the audio buffer,
- text hypothesis file, containing the context, decoder text buffer used for forced decoding, and the hypothesis.

The file structure may look like this:

```
seg_00002:
iter_00001_audio.wav  iter_00001_hypothesis.txt

seg_00004:
iter_00002_audio.wav       iter_00003_audio.wav       iter_00004_audio.wav       iter_00005_audio.wav
iter_00002_hypothesis.txt  iter_00003_hypothesis.txt  iter_00004_hypothesis.txt  iter_00005_hypothesis.txt

...
```

The hypotheses files may contain:

```
==> seg_00002/iter_00001_hypothesis.txt <==
CONTEXT+FORCED:	<|startoftranscript|><|en|><|transcribe|><|notimestamps|>
HYPOTHESIS:	 And so

==> seg_00004/iter_00002_hypothesis.txt <==
CONTEXT+FORCED:	<|startoftranscript|><|en|><|transcribe|><|notimestamps|>
HYPOTHESIS:	 And so,

==> seg_00004/iter_00003_hypothesis.txt <==
CONTEXT+FORCED:	<|startoftranscript|><|en|><|transcribe|><|notimestamps|> And so,
HYPOTHESIS:	 my fellow Americans

==> seg_00004/iter_00004_hypothesis.txt <==
CONTEXT+FORCED:	<|startoftranscript|><|en|><|transcribe|><|notimestamps|> And so, my fellow Americans
HYPOTHESIS:	, ask

==> seg_00004/iter_00005_hypothesis.txt <==
CONTEXT+FORCED:	<|startoftranscript|><|en|><|transcribe|><|notimestamps|> And so, my fellow Americans, ask
HYPOTHESIS:	 not.
```

Note that the very first segment and hypothesis `seg_00002/iter_00001_hypothesis.txt` is from "warm-up" processing. Before beginning of each compuatationally aware simulation, the first 1 second is processed by model so that the following updates are faster.


### Usage: Server -- real-time from mic 

The entry point `simulstreaming_canary_server.py` has the same model options as `simulstreaming_canary.py`, plus:
- `--host` and `--port` of the TCP connection, 
- `--warmup-file`: the warmup audio file is decoded by the Whisper backend after the model is loaded because without that, processing of the very the first input chunk may take longer.

See the help message (`-h` option).

Only computationally aware simulation is available with server. The option `--out-txt` produces only 2 timestamps columns, the beginning and end timestamps in the input audio. Emission time is not available.

**Linux** client example:

```
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

- `arecord` sends realtime audio from a sound device (e.g. mic), in raw audio format -- 16000 sampling rate, mono channel, S16_LE -- signed 16-bit integer low endian. (Or other operating systems, use another alternative)

- nc is netcat with server's host and port

**Windows/Mac**: `ffmpeg` may substitute `arecord`. Or use the solutions proposed in Whisper-Streaming pull requests [#111](https://github.com/ufal/whisper_streaming/pull/111) and [#123](https://github.com/ufal/whisper_streaming/pull/123).


### Usage: As a module

Analogically to using [WhisperStreaming as a module](https://github.com/ufal/whisper_streaming?tab=readme-ov-file#as-a-module).

**ELITR**:

SimulStreaming is one of follow up projects of [ELITR](https://elitr.eu) (European Live Translator, 2019-2022). It is designed as one of the tools in a complex distributed pipeline for long-form monologue speech transcription and translation between many languages (ref. to papers: [1](https://aclanthology.org/2020.iwltp-1.7/),[2](https://aclanthology.org/2021.mtsummit-asltrw.3/),[3](https://aclanthology.org/2021.eacl-demos.32/)). The other tools usable in such pipelines, as well as in other projects, are e.g. [Pipeliner](https://github.com/ELITR/pipeliner), [MT-wrapper](https://github.com/ELITR/mt-wrapper/), a front-end web application [online-text-flow](https://github.com/ELITR/online-text-flow), or a tool for [ASR latency evaluation](github.com/ufal/asr_latency).

## 📣 Feedback Welcome!

We, the authors of SimulStreaming from Charles University, are committed to
improving our research and the tool itself. Your experience as a user is
invaluable to us --- it can help to shape upcoming features, licensing models, and support services. 

To better understand your needs and guide the future of
SimulStreaming, we kindly ask the users, especially commercial, to fill out this **[questionnaire](https://forms.cloud.microsoft/e/7tCxb4gJfB).**

## 📄 Licence

MIT.

## 🤝 Contributions

Contributions welcome. 

A remarkable contribution project that integrates SimulStreaming is [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit).

## ✉️ Contact

[Dominik Macháček](https://ufal.mff.cuni.cz/dominik-machacek/), machacek@ufal.mff.cuni.cz

