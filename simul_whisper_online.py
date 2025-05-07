#!/usr/bin/env python3
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


from online_processor_interface import OnlineProcessorInterface

class OnlineASRProcessor(OnlineProcessorInterface):

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # there is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        
        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it
        
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            #k = len(self.commited)-1
            #while k>0 and self.commited[k][1] > l:
            #    k -= 1
            #t = self.commited[k][1] 
            logger.debug("chunking segment")
            #self.chunk_at(t)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []: return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []: return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:

            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")





    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)

class VACOnlineASRProcessor(OnlineProcessorInterface):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller). 

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds), 
    it runs VAD and continuously detects whether there is speech or not. 
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(self, online_chunk_size, online):
        self.online_chunk_size = online_chunk_size

        self.online = online

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use the default options there: 500ms silence, 100ms padding, etc.  

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)


    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]-self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"]-self.buffer_offset
                end = res["end"]-self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM. 
                self.buffer_offset += max(0,len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]


    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret



def common_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, 
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter '
                        'time, it waits, otherwise it processes the whole segment that was received by this time.')


    parser.add_argument('--lan', '--language', type=str, default='auto', 
                        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    parser.add_argument('--task', type=str, default='transcribe', 
                        choices=["transcribe","translate"],
                        help="Transcribe or translate.")

    parser.add_argument('--vac', action="store_true", default=False, 
                        help='Use VAC = voice activity controller. Recommended. Requires torch.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, 
                        help='VAC sample size in seconds.')
    parser.add_argument('--vad', action="store_true", default=False, 
                        help='Use VAD = voice activity detection, with the default parameters.')

    parser.add_argument("-l", "--log-level", dest="log_level", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Set the log level", default='DEBUG')


    


def asr_factory(args, backend=None, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    if backend is None:
        backend = args.backend
    if backend == "simul-whisper":
        from simul_whisper_backend import simul_asr_factory
        asr, online = simul_asr_factory(args, logfile=logfile)

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, online)

    if args.task == "translate":
        asr.set_translate_task()

    return asr, online

def set_logging(args,logger,other="_server"):
    logging.basicConfig(#format='%(name)s 
            format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online"+other).setLevel(args.log_level)
#    logging.getLogger("whisper_online_server").setLevel(args.log_level)


def main(entrypoint="simulwhisper"):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    common_args(parser)
    if entrypoint == "simulwhisper":
        from simul_whisper_backend import simulwhisper_args
        simulwhisper_args(parser)
        backend = "simul-whisper"
    else:
        raise ValueError(f"Unknown entrypoint: {entrypoint}")
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    
    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

#    if args.log_level:
#        logging.basicConfig(format='whisper-%(levelname)s:%(name)s: %(message)s',
#                            level=getattr(logging, args.log_level))

    set_logging(args,logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, backend, logfile=logfile)
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

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        else:
            # No text, so no output
            pass

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

            logger.debug(f"## last processed {end:.2f}s")

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
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    print("tady",o,file=sys.stderr)
    output_transcript(o, now=now)

if __name__ == "__main__":
    main(entrypoint="simulwhisper")

