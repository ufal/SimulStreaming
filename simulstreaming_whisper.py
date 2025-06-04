from whisper_streaming.base import OnlineProcessorInterface, ASRBase
import argparse

import sys
import torch

from simul_whisper.config import AlignAttConfig
from simul_whisper.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
from simul_whisper.whisper import tokenizer

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
    group.add_argument('--audio_min_len', type=float, default=3.0, 
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=24, 
                        help='Threshold for the attention-guided decoding, in frames.')

    group = parser.add_argument_group('Simul-Whisper\'s end-of-word detection')
    group.add_argument('--if_ckpt_path', type=str, default=None, 
                        help='The file path to the Simul-CIF checkpoint model. Align with the Whisper model, e.g., using small.pt for Whisper small.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False, help="Override CIF model. If True, CIF model will never fire. If False: if CIF model path is set, it triggers it, otherwise it always fires.")
#    group.add_argument('--segment_length', type=float, default=0.1, help='Chunk length, in seconds.')

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=None, help="Max context tokens for the model.")


def simul_asr_factory(args, logfile=sys.stderr):
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
    
    a = { v:getattr(args, v) for v in ["model_path", "if_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens'
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")
    print("INFO\tArguments:\n", a,file=sys.stderr)
    asr = SimulWhisperASR(**a,logfile=logfile)
    return asr, SimulWhisperOnline(asr, logfile)

class SimulWhisperASR(ASRBase):
    
    sep = " "

    def __init__(self, language, model_path, if_ckpt_path, frame_threshold, audio_max_len, audio_min_len, segment_length, beams, task, 
                 decoder_type, never_fire, init_prompt, static_init_prompt, max_context_tokens, logfile=sys.stderr):
        self.logfile = logfile
        cfg = AlignAttConfig(
            model_path=model_path, 
            segment_length=segment_length,
            frame_threshold=frame_threshold,
            language=language,
            audio_max_len=audio_max_len, 
            audio_min_len=audio_min_len,
            if_ckpt_path=if_ckpt_path,
            decoder_type=decoder_type, #"greedy" if beams==1 else "beam",
            beam_size=beams,
            task=task,
            never_fire=never_fire,
            init_prompt=init_prompt,
            max_context_tokens=max_context_tokens,
            static_init_prompt=static_init_prompt,
        )
        print(language,file=sys.stderr)
        self.model = PaddedAlignAttWhisper(cfg)

    def transcribe(self, audio, init_prompt=""):
        x = self.model.infer(audio, init_prompt=init_prompt)
        print(x,file=sys.stderr)
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
    def __init__(self, asr, logfile=sys.stderr):
        self.logfile = logfile
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
        
        print(result,file=sys.stderr)
        print("last attend frame",self.model.last_attend_frame,file=sys.stderr)
        print(self.model.max_text_len, file=sys.stderr)
        for i in range(3): print(file=sys.stderr)
        return (b,e,result)

    def finish(self):
        print("FINISH",file=sys.stderr)
        self.is_last = True
        #self.insert_audio_chunk(np.array([],dtype=np.float32))
        o = self.process_iter()
        self.is_last = False
        return o
    

if __name__ == "__main__":

    from whisper_streaming.whisper_online_main import main_simulation_from_file
    main_simulation_from_file(simul_asr_factory, add_args=simulwhisper_args)