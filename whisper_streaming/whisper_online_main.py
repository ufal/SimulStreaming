#!/usr/bin/env python3

# This code is retrieved from the original WhisperStreaming whisper_online.py .
# It is refactored and simplified. Only the code that is needed for the 
# SimulWhisper backend is kept. 


import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging


logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

def processor_args(parser):
    """shared args for the online processors
    parser: argparse.ArgumentParser object
    """
    group = parser.add_argument_group("WhisperStreaming processor arguments (shared for simulation from file and for the server)")
    group.add_argument('--min-chunk-size', type=float, default=1.2, 
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter '
                        'time, it waits, otherwise it processes the whole segment that was received by this time.')

    group.add_argument('--lan', '--language', type=str, default="en", 
                        help="Source language code, e.g. en, de, cs, or auto for automatic language detection from speech.")
    group.add_argument('--task', type=str, default='transcribe', 
                        choices=["transcribe","translate"],
                        help="Transcribe or translate.")

    group.add_argument('--vac', action="store_true", default=False, 
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    group.add_argument('--vac-chunk-size', type=float, default=0.04, 
                        help='VAC sample size in seconds.')

    parser.add_argument("-l", "--log-level", dest="log_level", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the log level", default='DEBUG')

    parser.add_argument("--logdir", help="Directory to save audio segments and generated texts for debugging.",
                       default=None)

def asr_factory(args, factory=None):
    """
    Creates and configures an asr and online processor object through factory that is implemented in the backend.
    """
#    if backend is None:
#        backend = args.backend
#    if backend == "simul-whisper":
#        from simul_whisper_backend import simul_asr_factory
    asr, online = factory(args)

    # Create the OnlineASRProcessor
    if args.vac:
        from whisper_streaming.vac_online_processor import VACOnlineASRProcessor
        online = VACOnlineASRProcessor(args.min_chunk_size, online)

    if args.task == "translate":
        if args.model_path.endswith(".en.pt"):
            logger.error(f"The model {args.model_path} is English only. Translation is not available. Terminating.")
            sys.exit(1)
        asr.set_translate_task()

    return asr, online

def set_logging(args,logger):
    logging.basicConfig(
        # this format would include module name:
        #    format='%(levelname)s\t%(name)s\t%(message)s')
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("simul_whisper").setLevel(args.log_level)
    logging.getLogger("whisper_streaming").setLevel(args.log_level)


def simulation_args(parser):
    simulation_group = parser.add_argument_group("Arguments for simulation from file")
    simulation_group.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    simulation_group.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    # TODO: offline mode is not implemented in SimulStreaming yet
#    simulation_group.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    simulation_group.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')

def main_simulation_from_file(factory, add_args=None):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''

    import argparse
    parser = argparse.ArgumentParser()

    processor_args(parser)
    if add_args is not None:
        add_args(parser)

    simulation_args(parser)

    args = parser.parse_args()
    args.offline = False  # TODO: offline mode is not implemented in SimulStreaming yet

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    set_logging(args,logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, factory)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path,0,1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    asr.warmup(a)

    beg = args.start_at
    start = time.time()-beg

    def output_transcript(iteration_output, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start

        if iteration_output:
            start_ts = iteration_output['start']
            end_ts = iteration_output['end']
            text = iteration_output['text']
            logger.debug(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}")
            print(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}", flush=True)
        else:
            logger.debug("No text in this segment")

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.info(f"## last processed {end:.2f}s")

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.info(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)