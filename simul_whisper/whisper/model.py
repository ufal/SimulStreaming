import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


# class LayerNorm(nn.LayerNorm):
#     def forward(self, x: Tensor) -> Tensor:
#         return super().forward(x.float()).type(x.dtype)

# class Linear(nn.Linear):
#     def forward(self, x: Tensor) -> Tensor:
#         return F.linear(
#             x,
#             self.weight.to(x.dtype),
#             None if self.bias is None else self.bias.to(x.dtype),
#         )


# class Conv1d(nn.Conv1d):
#     def _conv_forward(
#         self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
#     ) -> Tensor:
#         return super()._conv_forward(
#             x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
#         )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

import sys  ## this is mine, for debugging
class MultiHeadAttention(nn.Module):

    use_sdpa = False  # disabling: https://github.com/linto-ai/whisper-timestamped/issues/212

    def __init__(self, n_state: int, n_head: int, cache_id: str):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.key.cache_id = f"{cache_id}_key"
        self.value = nn.Linear(n_state, n_state)
        self.value.cache_id = f"{cache_id}_value"
        self.out = nn.Linear(n_state, n_state)
        self.cache_id = cache_id

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        #print("MultiHeadAttention forward",file=sys.stderr)
        q = self.query(x)
#        print(q.shape, x is None, mask is None, list(kv_cache.keys()) if kv_cache is not None else None, file=sys.stderr)
        # print(mask, kv_cache, xa, file=sys.stderr)

        if kv_cache is None or xa is None or self.key.cache_id not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
            # print(self.key.cache_id, "cache miss") # , kv_cache is None, xa is None, self.key.cache_id not in kv_cache if kv_cache is not None else None, k.shape, x.shape)
            # if kv_cache is not None:
            #     print(kv_cache.keys())
        else:
            # print(self.key.cache_id, "cache hit") #, kv_cache is None, xa is None, self.key.cache_id not in kv_cache)
            # if kv_cache is not None:
            #     print(kv_cache.keys())
            k = kv_cache[self.key.cache_id]
            v = kv_cache[self.value.cache_id]
        # print(self.key.cache_id, "qkv attention", q.shape, k.shape, v.shape)
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    # def qkv_attention(
    #     self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    # ):
    #     n_batch, n_ctx, n_state = q.shape
    #     scale = (n_state // self.n_head) ** -0.25
    #     q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    #     k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    #     v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    #     qk = q @ k
    #     if mask is not None:
    #         qk = qk + mask[:n_ctx, :n_ctx]
    #     # qk = qk.float()

    #     w = F.softmax(qk, dim=-1) # .to(q.dtype)
    #     return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cache_id: str="", cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, cache_id=f"{cache_id}_self_attn")
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head, cache_id=f"{cache_id}_cross_attn") if cross_attention else None

        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # print("ResidualAttentionBlock forward",file=sys.stderr)
        # print(x.shape, file=sys.stderr)
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cache_id=f"enc_layer{i}") for i in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor, return_layer_results: bool=False):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1) # BDT -> BTD

        # 两层卷积，2倍降采样
        # 最终剩下1500帧

        x = (x + self.positional_embedding[:x.shape[1], :]) #.to(x.dtype)

        layer_results = []
        i = 0
        for block in self.blocks:
            # print(f"encoder layer {i}")
            x = block(x)
            layer_results.append(x)
            i += 1

        x = self.ln_post(x)

        if return_layer_results:
            return x, layer_results
        else:
            return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, cache_id=f"dec_layer{i}")
                for i in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # x = x.to(xa.dtype)

        i = 0
        for block in self.blocks:
            # print(f"decoder layer {i}")
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            i += 1

        x = self.ln(x)
        logits = x @ torch.transpose(self.token_embedding.weight, 0, 1)

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        # tokens = tokens.to(self.decoder.ln.weight.dtype)
        # audio_features = audio_features.to(self.decoder.ln.weight.dtype)
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # mel = mel.to(self.decoder.ln.weight.dtype)
        # tokens = tokens.to(self.decoder.ln.weight.dtype)
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    # 为decoder加入缓存机制，每次推理时保存上次的k和v，下次推理无需重新计算
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
