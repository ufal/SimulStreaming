from whisper_streaming.base import OnlineProcessorInterface, ASRBase
import argparse

import sys
import logging
import torch

from simul_whisper.config import AlignAttConfig
from simul_whisper.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
from simul_whisper.whisper import tokenizer

logger = logging.getLogger(__name__)


def simulwhisper_args(parser):
    group = parser.add_argument_group('Whisper arguments')
    group.add_argument('--model_path', type=str, default='./large-v3.pt', 
                        help='The file path to the Whisper .pt model. If not present on the filesystem, the model is downloaded automatically.')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.")
    group.add_argument("--decoder",type=str, default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=30.0, 
                        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--audio_min_len', type=float, default=0.0, 
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=25, 
                        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
                            'until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model. ')

    group = parser.add_argument_group('Truncation of the last decoded word (from Simul-Whisper)')
    group.add_argument('--cif_ckpt_path', type=str, default=None, 
                        help='The file path to the Simul-Whisper\'s CIF model checkpoint that detects whether there is' \
                        'end of word at the end of the chunk. If not, the last decoded space-separated word is truncated ' \
                        'because it is often wrong -- transcribing a word in the middle.' \
                        'The CIF model adapted for the Whisper model version should be used. ' \
                        'Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . ' \
                        'Note that there is no model for large-v3.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False, 
                       help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. " \
                       ". If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. " \
                        "Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.")

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=None, help="Max context tokens for the model. Default is 0.")


def simul_asr_factory(args):
    decoder = args.decoder
    if args.beams > 1:
        if decoder == "greedy":
            raise ValueError("Invalid 'greedy' decoder type for beams > 1. Use 'beam'.")
        elif decoder is None or decoder == "beam":
            decoder = "beam"
        else:
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
    else:
        if decoder is None:
            decoder = "greedy"
        elif decoder not in ("beam","greedy"):
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
        # else: it is greedy or beam, that's ok 
    
    a = { v:getattr(args, v) for v in ["model_path", "cif_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens'
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")
    logger.info(f"Arguments: {a}")
    asr = SimulWhisperASR(**a)
    return asr, SimulWhisperOnline(asr)

class SimulWhisperASR(ASRBase):
    
    sep = " "

    def __init__(self, language, model_path, cif_ckpt_path, frame_threshold, audio_max_len, audio_min_len, segment_length, beams, task, 
                 decoder_type, never_fire, init_prompt, static_init_prompt, max_context_tokens):
        cfg = AlignAttConfig(
            model_path=model_path, 
            segment_length=segment_length,
            frame_threshold=frame_threshold,
            language=language,
            audio_max_len=audio_max_len, 
            audio_min_len=audio_min_len,
            cif_ckpt_path=cif_ckpt_path,
            decoder_type=decoder_type, #"greedy" if beams==1 else "beam",
            beam_size=beams,
            task=task,
            never_fire=never_fire,
            init_prompt=init_prompt,
            max_context_tokens=max_context_tokens,
            static_init_prompt=static_init_prompt,
        )
        logger.info(f"Language: {language}")
        self.model = PaddedAlignAttWhisper(cfg)

    def transcribe(self, audio, init_prompt=""):
        x = self.model.infer(audio, init_prompt=init_prompt)
        logger.debug(f"Transcription: {x}")
        return x

    def warmup(self, audio, init_prompt=""):
        self.model.infer(audio, True)
        self.model.refresh_segment(complete=True)
    
    def use_vad(self):
        print("VAD not implemented",file=sys.stderr)

    def set_translate_task(self):
        self.model.tokenizer = tokenizer.get_tokenizer(multilingual=True, language=self.model.cfg.language, 
                                                             num_languages=self.model.model.num_languages,
                                                             task="translate")


class SimulWhisperOnline(OnlineProcessorInterface):
    def __init__(self, asr):
        self.model = asr.model
        self.init()

    def init(self, offset=None):
        self.audio_chunks = []
        if offset is not None:
            self.offset = offset
        else:
            self.offset = 0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset

    def insert_audio_chunk(self, audio):
        self.audio_chunks.append(torch.from_numpy(audio))

    def process_iter(self):
        if len(self.audio_chunks) == 0:
            audio = None
        else:
            audio = torch.cat(self.audio_chunks, dim=0)
            if audio.shape[0] == 0:
                audio = None
            else:
                self.end += audio.shape[0] / self.SAMPLING_RATE #self.model.cfg.segment_length
        self.audio_chunks = []
        #print("audio shape",audio.shape,flush=True,file=sys.stderr)
#        print((len(self.model.segments)+1) * self.model.cfg.segment_length, self.model.cfg.buffer_len)

#        n = self.model.infer(audio,"",force_decode="",is_last=self.is_last)
        n = self.model.infer(audio,is_last=self.is_last)

        #print("tady",n,file=sys.stderr)
        n = n[n<DEC_PAD]
        #print("OUTPUT <DEC_PAD",n,file=sys.stderr)
#        result = n
        result = self.model.tokenizer.decode(n)
        #print("RESULT",result,file=sys.stderr)
        if len(result) == 0:
            return (None,None,"")
        b = self.beg
        e = self.end
        self.beg = self.end
        
        logger.debug(f"Result: {result}")
        logger.debug(f"Last attend frame: {self.model.last_attend_frame}")
        logger.debug(f"Max text length: {self.model.max_text_len}")
        return (b,e,result)

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        #self.insert_audio_chunk(np.array([],dtype=np.float32))
        o = self.process_iter()
        self.is_last = False
        return o
    

if __name__ == "__main__":

    from whisper_streaming.whisper_online_main import main_simulation_from_file
    main_simulation_from_file(simul_asr_factory, add_args=simulwhisper_args)