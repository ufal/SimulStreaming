import os
import logging

import torch
import torch.nn.functional as F

from ..whisper import load_model, DecodingOptions, tokenizer
from .config import AlignAttConfig
from ..whisper.audio import log_mel_spectrogram, TOKENS_PER_SECOND, pad_or_trim, N_SAMPLES, N_FRAMES
from ..whisper.timing import median_filter
from ..whisper.decoding import SuppressBlank, GreedyDecoder, BeamSearchDecoder, SuppressTokens, PyTorchInference
import os

from token_buffer import TokenBuffer

import numpy as np

DEC_PAD = 50257
logger = logging.getLogger(__name__)

import sys
def dprint(a):
    print("simul_whisper DEBUG",a,file=sys.stderr)
logger.debug = dprint

class PaddedAlignAttWhisper:
    def __init__(self, cfg: AlignAttConfig) -> None:
        self.debug_iterations = 0
        model_name = os.path.basename(cfg.model_path).replace(".pt", "")
        model_path = os.path.dirname(cfg.model_path)
        self.model = load_model(name=model_name, download_root=model_path)

        print(self.model.dims,file=sys.stderr)


        decode_options = DecodingOptions(
            language = cfg.language, 
            without_timestamps = True,
            task=cfg.task
        )
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=True, 
            language=cfg.language, 
            num_languages=self.model.num_languages,
            task=decode_options.task
        )
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)
        self.cfg = cfg


        self.CIFLinear = torch.nn.Linear(self.model.dims.n_audio_state, 1)
        # if cfg.never_fire == True:
        #     self.never_fire = True
        #     self.always_fire = False
        if cfg.if_ckpt_path is None or not cfg.if_ckpt_path:
            if cfg.never_fire:
                self.never_fire = True
                self.always_fire = False
            else:
                self.always_fire = True
                self.never_fire = False
        else:
            self.always_fire = False
            self.never_fire = cfg.never_fire
            checkpoint = torch.load(cfg.if_ckpt_path)
            self.CIFLinear.load_state_dict(checkpoint)
        self.CIFLinear.to(self.model.device)
        # install hooks
        self.dec_attns = []
        def layer_hook(module, net_input, net_output):
            # net_output[1]: B*num_head*token_len*audio_len
            t = F.softmax(net_output[1], dim=-1)
            self.dec_attns.append(t.squeeze(0))
        for b in self.model.decoder.blocks:
            b.cross_attn.register_forward_hook(layer_hook)
        
        self.kv_cache = {}
        def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
#            return net_output
            #print(net_output.shape, "net_output shape", file=sys.stderr)
            if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
                # save as-is, for the first token or cross attention
                self.kv_cache[module.cache_id] = net_output
            else:
                x = self.kv_cache[module.cache_id]
            #    print(x.shape, "x shape", file=sys.stderr)
            #    print(module.cache_id, file=sys.stderr)
            #    print(module, file=sys.stderr)
                self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
            return self.kv_cache[module.cache_id] 

        for i,b in enumerate(self.model.decoder.blocks):
            b.attn.key.register_forward_hook(kv_hook)
            b.attn.value.register_forward_hook(kv_hook)
            b.cross_attn.key.register_forward_hook(kv_hook)
            b.cross_attn.value.register_forward_hook(kv_hook)

        self.align_source = {}
        self.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.align_source.get(layer_rank, [])
            heads.append((self.num_align_heads, head_id.item()))
            self.align_source[layer_rank] = heads
            self.num_align_heads += 1

        self.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence_including_notimestamps, 
            dtype=torch.long, 
            device=self.model.device).unsqueeze(0)
        self.initial_token_length = self.initial_tokens.shape[1]
        self.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)

        suppress_tokens = [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
                # self.tokenizer.eot 
                self.tokenizer.no_timestamps,  # added by DM
            ] + list(self.tokenizer.all_language_tokens)  # added by DM
            #self.tokenizer.special_tokens
        #suppress_tokens = [v for v in self.tokenizer.special_tokens.values() if v != self.tokenizer.eot]
        #print(sorted(self.tokenizer.special_tokens.values()), file=sys.stderr)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens =  tuple(sorted(set(suppress_tokens)))
        print(suppress_tokens, file=sys.stderr)

        def suppress_blank_logits(logits, tokens, init_len):
            f = SuppressBlank(self.tokenizer, init_len)
            return f.apply(logits, tokens)
        self.suppres_blank = suppress_blank_logits
        sup_tokens = SuppressTokens(suppress_tokens)
        self.suppress_tokens = lambda logits: sup_tokens.apply(logits, None)
#        self.logit_filters = [
#            SuppressBlank(self.tokenizer, self.initial_token_length),
#            SuppressTokens(suppress_tokens)
#            ]
        if cfg.decoder_type == "greedy":
            print("greedy decoder",file=sys.stderr)
            self.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)
            self.decoder_type = "greedy"

        elif cfg.decoder_type == "beam":
            self.decoder_type = "beam"

            class MyPyTorchInference(PyTorchInference):

                def _kv_modules(self):
                    key_modules = [block.attn.key.cache_id for block in self.model.decoder.blocks]
                    value_modules = [block.attn.value.cache_id for block in self.model.decoder.blocks]
                    return key_modules + value_modules


                # def __init__(self, model, initial_token_length, kv_cache):
                #     super().__init__(model, initial_token_length)
                #     self.kv_cache = {}
                def rearrange_kv_cache(self, source_indices):
                    #return
                    #source_indices = source_indices[:1]
                    #print("source_indices",source_indices, file=sys.stderr)
                    if source_indices != list(range(len(source_indices))):
                        #print("tady",file=sys.stderr)
                        for module_cache_id in self._kv_modules():
                            # update the key/value cache to contain the selected sequences
                            #print("before",self.kv_cache[module].shape, file=sys.stderr)
#                            print("before",self.kv_cache[module], file=sys.stderr)

                            self.kv_cache[module_cache_id] = self.kv_cache[module_cache_id][source_indices].detach()
                            #print("after",self.kv_cache[module].shape, file=sys.stderr)
                            #print("after",self.kv_cache[module], file=sys.stderr)

                    # else:
                    #     print("HHHHHHHHHHHHHHHHHHHHHHHHHERE", file=sys.stderr)
                from torch import Tensor
                def logits(self, tokens: Tensor, audio_features: Tensor, offset: int) -> Tensor:
                    # if not self.kv_cache:
                    #     self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        #            print(tokens.shape, "tokens shape", file=sys.stderr)
        #            print(audio_features.shape, "audio_features shape", file=sys.stderr)
                    return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache, offset=offset)

            self.inference = MyPyTorchInference(self.model, self.initial_token_length)
            self.inference.kv_cache = self.kv_cache

            class MyBeamSearchDecoder(BeamSearchDecoder):
                def update(self, tokens, logits, sum_logprobs):
                    #tokens = tokens.repeat(self.beam_size,1)
                    # print(logits.shape, file=sys.stderr)
                    # sum_logprobs = sum_logprobs.repeat(self.beam_size)
                    # logits = logits.repeat(self.beam_size,1)
                    return super().update(tokens, logits, sum_logprobs)

            self.token_decoder = MyBeamSearchDecoder(inference=self.inference, eot=self.tokenizer.eot, beam_size=cfg.beam_size)
            print("EOT token is:",self.tokenizer.eot, file=sys.stderr)

        # init state
        self.segments = []
        self.tokens = [self.initial_tokens]
        self.last_attend_frame = -self.cfg.rewind_threshold

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens
        self.init_context()

    def init_context(self):
        kw = {'tokenizer': self.tokenizer, 
              'device': self.model.device, 
              'prefix_token_ids': [self.tokenizer.sot_prev]}
        self.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.context.text += self.cfg.init_prompt

    def trim_context(self):
        print("TRIM CONTEXT",file=sys.stderr)
        c = len(self.context.as_token_ids()) - len(self.context.prefix_token_ids)
        print("c=", len(self.context.as_token_ids()), len(self.context.prefix_token_ids), file=sys.stderr)
        print(self.context.as_text(), file=sys.stderr)
        print(self.context.as_tensor(), file=sys.stderr)
        l = sum(t.shape[1] for t in self.tokens) + c
        print("len", l, c, self.max_context_tokens, file=sys.stderr)
        if self.cfg.static_init_prompt is None:
            after = 0
        else:
            after = len(self.cfg.static_init_prompt)
        print("len", l, c, self.max_context_tokens, file=sys.stderr, flush=True)
        while c > self.max_context_tokens or l > self.max_text_len - 20:
            t = self.context.trim_words(after=after)
            l -= t
            c -= t
            print("len", l, c, self.max_context_tokens, file=sys.stderr, flush=True)
            if t == 0:
                break
        print("len", l, c, self.max_context_tokens, file=sys.stderr)
        print("CONTEXT AFTER TRIM:", self.context.text, "len", l, file=sys.stderr)



    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor, offset: int) -> torch.Tensor:
        if self.cfg.decoder_type == "greedy":
            logit = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
        else:
            print("LOGITS",tokens.shape, file=sys.stderr)
           # if tokens.shape[0] == 1:
           #     toks = tokens.repeat_interleave(self.cfg.beam_size, dim=0).to(audio_features.device)
           # else:
            toks = tokens
            logit = self.inference.logits(toks, audio_features, offset)
        return logit
    

    def refresh_segment(self, complete=False):

        print("refreshing segment",file=sys.stderr)
        self.tokens = [self.initial_tokens] 
        self.last_attend_frame = -self.cfg.rewind_threshold       
        self.init_context()
        print("CONTEXT", self.context, file=sys.stderr)
        if not complete and len(self.segments) > 2:
            print("jsme tady",file=sys.stderr)
            self.segments = self.segments[-2:]
        else:
            print("nejsme tady",file=sys.stderr)
            self.segments = []


    # from https://github.com/dqqcasia/mosst/blob/master/fairseq/models/speech_to_text/convtransformer_wav2vec_cif.py
    def resize(self, alphas, target_lengths, threshold=0.999):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        # sum
        _num = alphas.sum(-1)
        num = target_lengths.float()
        # scaling
        _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))
        # rm attention value that exceeds threashold
        count = 0
        while len(torch.where(_alphas > threshold)[0]):
            count += 1
            if count > 10:
                break
            print('fixing alpha',file=sys.stderr)
            xs, ys = torch.where(_alphas > threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num   
    
    
    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        if self.always_fire: return True
        if self.never_fire: return False
        content_mel_len = chunked_encoder_feature.shape[1] # B, T, D
        alphas = self.CIFLinear(chunked_encoder_feature).squeeze(dim=2) # B, T
        alphas = torch.sigmoid(alphas)
        decode_length = torch.round(alphas.sum(-1)).int()
        alphas, _ = self.resize(alphas, decode_length)
        alphas = alphas.squeeze(0) # (T, )
        threshold = 0.999
        integrate = torch.cumsum(alphas[:-1], dim=0) # ignore the peak value at the end of the content chunk
        exceed_count = integrate[-1] // threshold
        integrate = integrate - exceed_count*1.0 # minus 1 every time intergrate exceed the threshold
        important_positions = (integrate >= 0).nonzero(as_tuple=True)[0]
        if important_positions.numel() == 0:
            return False
        else:
            return important_positions[0] >= content_mel_len-2

    def segments_len(self):
        segments_len = sum(s.shape[0] for s in self.segments) / 16000
        return segments_len


    def _apply_minseglen(self):
        segments_len = self.segments_len()
        # wait for long enough audio to start
        if segments_len < self.cfg.min_seg_len: 
            logger.debug("waiting for next segment")
            return False
        # len of audio is bigger than buffer_len. Going to remove the first segment
        while segments_len > self.cfg.buffer_len:
            removed_len = self.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.last_attend_frame -= int(TOKENS_PER_SECOND*removed_len)
            self.segments = self.segments[1:]
            logger.debug(f"remove segments: {len(self.segments)} {len(self.tokens)}")
        
            self.context.append_token_ids(self.tokens[1][0,:])
            self.tokens = [self.initial_tokens] + self.tokens[2:]
        return True

    def _current_tokens(self):

        print("self.tokens in current_tokens",self.tokens, file=sys.stderr)
        toks = self.tokens
        # very first infer: duplicate start of seq to beam_size
        if toks[0].shape[0] == 1:
            toks[0] = toks[0].repeat_interleave(self.cfg.beam_size,dim=0)

        if not self.context.is_empty():
#             context_text = self.context.as_text()
#             print("CONTEXT TEXT:",context_text, file=sys.stderr)

#             context_toks = self.context_as_tok_ids(self.tokenizer, prefix=[self.tokenizer.sot_prev]) #.encode(context_text)
#             print("CONTEXT TOKENS", context_toks, file=sys.stderr)

# #            context_toks = torch.tensor(context_toks, dtype=torch.long, device=self.model.device).unsqueeze(0)
#             print("CONTEXT TOKENS", context_toks, file=sys.stderr)

            context_toks = self.context.as_tensor_beam(self.cfg.beam_size, device=self.model.device)
            print("CONTEXT TOKENS", context_toks, file=sys.stderr)

            toks = [context_toks] + toks
            print("toks with context",toks, file=sys.stderr)

        # make it one tensor
        if len(toks) > 1:
            #print("tokens:", toks, file=sys.stderr)
            current_tokens = torch.cat(toks, dim=1)
        else:
            print("current tokens: skipping cat", toks, file=sys.stderr)
            current_tokens = toks[0]
#        print("current_tokens", current_tokens, file=sys.stderr)
        self.debug_print_tokens(current_tokens)
        return current_tokens


    def debug_print_tokens(self, tokens):
        for i in range(self.cfg.beam_size):
            print(self.tokenizer.decode_with_timestamps(tokens[i].tolist(),), file=sys.stderr)

    @torch.no_grad()
    def infer(self, segment=None, is_last=False):
        new_segment = True
        if segment is not None:
            self.segments.append(segment)
        elif len(self.segments) == 0:
            return self.initial_tokens.new_tensor([])
        else:
            segment = self.segments[-1]
        if not self._apply_minseglen():
            return self.initial_tokens.new_tensor([])


        # input_segments is concatenation of audio, it's one array
        if len(self.segments) > 1:
            input_segments = torch.cat(self.segments, dim=0)
        else:
            input_segments = self.segments[0]

        self.trim_context()
        current_tokens = self._current_tokens()
        
        # mel + padding to 30s
        mel_padded = log_mel_spectrogram(input_segments, n_mels=self.model.dims.n_mels, padding=N_SAMPLES, 
                                            device=self.model.device).unsqueeze(0)
        logger.debug(f"after padding: {mel_padded.shape}")

        # trim to 3000
        mel = pad_or_trim(mel_padded, N_FRAMES)
        logger.debug(f"after trim {mel.shape}")
        # the len of actual audio
        content_mel_len = int((mel_padded.shape[2] - mel.shape[2])/2)

        encoder_feature = self.model.encoder(mel)
        sum_logprobs = torch.zeros(self.cfg.beam_size, device=mel.device)
        completed = False

        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])


        ####################### Decoding loop
        print("DECODING LOOP STARTS", file=sys.stderr)

        attn_of_alignment_heads = None
        most_attended_frame = None
        #print("len",current_tokens.shape[1], current_tokens.shape[1] < self.max_text_len,file=sys.stderr)

        token_len_before_decoding = current_tokens.shape[1]
        while not completed and current_tokens.shape[1] < self.max_text_len: # bos is 3 tokens
#        while current_tokens.shape[1] < 20: #self.max_text_len: # bos is 3 tokens

            if new_segment:
                tokens_for_logits = current_tokens
            else:
                # only need to use the last token except in the first forward pass
                tokens_for_logits = current_tokens[:,-1:]

            logits = self.logits(tokens_for_logits, encoder_feature, offset=current_tokens.shape[1]) # B, len(tokens), token dict size
#            print(self.kv_cache.keys(), file=sys.stderr)

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = logits[:, self.sot_index, :].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    logger.debug("no speech, stop")
                    break

            logits = logits[:, -1, :] # logits for the last token
            #self.suppres_blank(logits, current_tokens, token_len_before_decoding)
            if new_segment:
                logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            new_segment = False
            self.suppress_tokens(logits)
#            for logits_filter in self.logit_filters:  # the first filter is suppress blank. It is active only for the very first token after sot, when when current_tokens 
                # len is self.initial_token_length.
                # the second one suppressess many special tokens. It doesn't care about current_tokens.
#                logits_filter.apply(logits, current_tokens)

#            print("current_tokens_shape", current_tokens.shape, file=sys.stderr)
            #toks = current_tokens.repeat_interleave(self.cfg.beam_size, dim=0).to(encoder_feature.device)
            print(logits.shape, "= logits shape", file=sys.stderr)
#            print("XXXXXXXXXXXXXXXXXXXXXX",file=sys.stderr)
            current_tokens, completed = self.token_decoder.update(current_tokens, logits, sum_logprobs)
#            print("YYYYYYYYYYYYYY",file=sys.stderr)
            print(current_tokens, completed, file=sys.stderr)
            self.debug_print_tokens(current_tokens)
            print(sum_logprobs, "= sum_logprobs", file=sys.stderr)
#            print(sum_logprobs.shape, "= sum_logprobs shape", file=sys.stderr)

#            print(current_tokens.shape, file=sys.stderr)
            if self.decoder_type == "beam":
                print(self.token_decoder.finished_sequences, file=sys.stderr)

                logprobs = F.log_softmax(logits.float(), dim=-1)
                idx = 0
                print("beam:",logprobs[idx].topk(self.cfg.beam_size + 1),file=sys.stderr)
                print("greedy:", logits.argmax(dim=-1),file=sys.stderr)
            if completed:
                self.debug_print_tokens(current_tokens)

                logger.debug("decode stopped because decoder completed")

            attn_of_alignment_heads = [[] for _ in range(self.num_align_heads)]
            for i, attn_mat in enumerate(self.dec_attns):
                layer_rank = int(i % len(self.model.decoder.blocks))
                align_heads_in_layer = self.align_source.get(layer_rank, [])
                if len(align_heads_in_layer) == 0:
                    continue
                for align_head_rank, head_id in align_heads_in_layer:
#                    if i == 0:
#                        print(head_id, attn_mat.shape, file=sys.stderr)
                    if self.cfg.beam_size == 1:
                        a = attn_mat[head_id, :, :]
                        a = a.unsqueeze(0)
                    else:
                        a = attn_mat[:, head_id, :, :]
                    # if i == 0:
                    #     print(align_head_rank, self.num_align_heads, file=sys.stderr)
                    attn_of_alignment_heads[align_head_rank].append(a)
            #print(attn_of_alignment_heads, file=sys.stderr)
            tmp = []
            for mat in attn_of_alignment_heads:
                t = torch.cat(mat, dim=1)# bylo tam dim=0
#                print(t.shape, file=sys.stderr)
                tmp.append(t) 
            attn_of_alignment_heads = torch.stack(tmp, dim=1)
            print(attn_of_alignment_heads.shape, "tttady", file=sys.stderr)
            std, mean = torch.std_mean(attn_of_alignment_heads, dim=-2, keepdim=True, unbiased=False)
            attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std
            attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7) # from whisper.timing
            attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=1)
            print(attn_of_alignment_heads.shape, "po mean", file=sys.stderr)
            attn_of_alignment_heads = attn_of_alignment_heads[:,:, :content_mel_len]
            print(attn_of_alignment_heads.shape, "pak ", file=sys.stderr)

            # for each beam, the most attended frame is:
            most_attended_frames = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)
            print(most_attended_frames.shape, "most att frames", file=sys.stderr)

#            most_attended_frames = most_attended_frames.squeeze(dim=0)
#            print(most_attended_frames.shape, "most att frames sq", file=sys.stderr)
            print(most_attended_frames, "most att f", file=sys.stderr)
            most_attended_frame = most_attended_frames[0].item()
            #most_attended_frame = torch.min(most_attended_frames).item()  # converting to scalar
            print("most att f", most_attended_frame, file=sys.stderr)

            print("current tokens", current_tokens.shape, file=sys.stderr)
            if completed:
            #    # stripping the last token, the eot
                current_tokens = current_tokens[:, :-1]
                break
            
            # for some rare cases where the attention fails
            if not is_last and self.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                # TODO: check this
                if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                    logger.debug("ommit rewinding from special tokens")
                    self.last_attend_frame = most_attended_frame
                else:
                    logger.debug(
                        f"[rewind detected] current attention pos: {most_attended_frame}, "
                        f"last attention pos: {self.last_attend_frame}; omit this segment")
                    self.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = torch.cat(self.tokens, dim=1) if len(self.tokens) > 0 else self.tokens[0]
                    break
            else:
                self.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (4 if is_last else self.cfg.frame_threshold):
                logger.debug(f"attention reaches the end: {most_attended_frame}/{content_mel_len}")
                # stripping the last token, the one that is attended too close to the end
                current_tokens = current_tokens[:, :-1]
                break
        
            # debug print
            for i in range(self.cfg.beam_size):
                logger.debug("attn: {}, current pos: {}, current token: {}({})".format(
                    attn_of_alignment_heads.shape if attn_of_alignment_heads is not None else None,
                    most_attended_frames[i], 
                    current_tokens[i, -1].item(),
                    self.tokenizer.decode([current_tokens[i, -1].item()])
                ))
        ####################### End of decoding loop

        print("END OF DECODING LOOP", file=sys.stderr)

        print("sum_logprobs", sum_logprobs, file=sys.stderr)
        if attn_of_alignment_heads is not None:
            seg_len = int(segment.shape[0] / 16000 * TOKENS_PER_SECOND)
            #int(self.cfg.segment_length*TOKENS_PER_SECOND)

            print(seg_len, self.cfg.segment_length, TOKENS_PER_SECOND, file=sys.stderr)
            print(attn_of_alignment_heads.shape,file=sys.stderr)
            print("token_len_before_decoding", token_len_before_decoding, file=sys.stderr)
            print(attn_of_alignment_heads[:, token_len_before_decoding:, -seg_len:].shape, file=sys.stderr)


            # we consider that the beam hypothesis are ordered from the best to the worst. 
            # the best has index 0
            print("sum_logprobs", sum_logprobs, file=sys.stderr)

            # Lets' now consider only the top hypothesis in the beam search
            top_beam_attn_of_alignment_heads = attn_of_alignment_heads[0]
            print(top_beam_attn_of_alignment_heads.shape, "top beam attn shape", file=sys.stderr)


            # debug print: how is the new token attended?
            new_token_attn = top_beam_attn_of_alignment_heads[token_len_before_decoding:, -seg_len:]
            print(new_token_attn.shape, "new token attn shape", file=sys.stderr)
            if new_token_attn.shape[0] == 0:  # it's not attended in the current audio segment
                logger.debug("no token generated")
                logger.debug(f"token len {current_tokens.shape}")
            else:  # it is, and the max attention is:
                new_token_max_attn, _ = new_token_attn.max(dim=-1)
                logger.debug(f"segment max attention: {new_token_max_attn.mean().item()/len(self.segments)}")

        # let's now operate only with the top beam hypothesis
        #new_hypothesis = current_tokens[0].new_tensor([]).unsqueeze(0)
        print("tokens_to_split",file=sys.stderr)
        tokens_to_split = current_tokens[0, token_len_before_decoding:]
        print(tokens_to_split.shape,file=sys.stderr)
        if fire_detected or is_last:
            new_hypothesis = tokens_to_split
        else:
            print(tokens_to_split, file=sys.stderr)
            print(tokens_to_split.shape, file=sys.stderr)
            tokens_to_split = tokens_to_split
            print(tokens_to_split.shape, file=sys.stderr)
            text_to_split = self.tokenizer.decode(tokens_to_split)
            print(text_to_split, file=sys.stderr)
            logger.debug("text at current step: {}".format(text_to_split.replace(" ", "<space>")))
            text_before_space = " ".join(text_to_split.split(" ")[:-1])
            logger.debug("before the last space: {}".format(text_before_space.replace(" ", "<space>")))
            if len(text_before_space) > 0:
                new_hypothesis = current_tokens.new(self.tokenizer.encode(text_before_space, 
                                                                          allowed_special="all"))
            else:
                new_hypothesis = current_tokens.new_tensor([])


### add hypothesis
        print("new_hypo", new_hypothesis, file=sys.stderr)
        ret = new_hypothesis.squeeze(0)
        ap = new_hypothesis.unsqueeze(0).repeat_interleave(self.cfg.beam_size, dim=0)
        self.tokens.append(ap.clone())
        print("ret", ret, file=sys.stderr)
        ret = ret[ret<DEC_PAD]
        print("ret", ret, file=sys.stderr)
        print("ap", ap, file=sys.stderr)
        
        # # Print self.tokens before appending
        # print("self.tokens before appending:", self.tokens, file=sys.stderr)
        
        
        # # Print self.tokens after appending
        # print("self.tokens after appending:", self.tokens, file=sys.stderr)

        self.dec_attns = []
        self.kv_cache = {}
        if self.decoder_type == "beam":
            self.inference.kv_cache = self.kv_cache
            self.token_decoder.reset()


        print("OUTPUT:",self.tokenizer.decode(ret),file=sys.stderr)
        for i in range(10):
            print(file=sys.stderr)
        

        # self.debug_iterations += 1
        # if self.debug_iterations > 5:
        #     sys.exit()
        return ret
