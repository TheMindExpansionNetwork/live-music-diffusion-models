from functools import reduce
import random
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from typing import Callable, Literal
from torch.nn.attention.flex_attention import flex_attention
import os

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

from .utils import compile

# dynamic=True allows varying Q/KV lengths across calls (e.g. block-causal bootstrap vs
# streaming first-step masks have different shapes). Without this, static-shape compilation
# replays CUDA graphs with wrong dimensions → illegal memory access.
flex_attention_compiled = torch.compile(flex_attention, dynamic=True)

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 1000000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward_from_positions(self, positions: torch.Tensor):
        """
        Compute rotary embeddings for explicit position indices.

        Args:
            positions: (seq_len,) or (batch, seq_len) integer or float position indices.

        Returns:
            freqs of shape (seq_len, d) when positions is 1-D, or
            (batch, seq_len, d) when positions is 2-D.
            Scale is computed per-position when use_xpos=True, otherwise 1.
        """
        device = self.inv_freq.device
        positions = positions.to(torch.float32).to(device) / self.interpolation_factor

        if positions.ndim == 1:
            freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            if self.scale is None:
                return freqs, 1.
            power = (positions - (positions.shape[0] // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)
            return freqs, scale
        else:
            # Batched: (batch, seq_len)
            freqs = torch.einsum('b i, j -> b i j', positions, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            if self.scale is None:
                return freqs, 1.
            # Use position values themselves (not seq indices) for xpos scale
            power = (positions - (positions.shape[1] // 2)) / self.scale_base  # (batch, seq_len)
            scale = self.scale.unsqueeze(0).unsqueeze(0) ** rearrange(power, 'b n -> b n 1')  # (batch, seq_len, d/2)
            scale = torch.cat((scale, scale), dim=-1)  # (batch, seq_len, d)
            return freqs, scale

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    # Trim freqs to the current sequence length.
    # freqs can be 2-D (seq, d) for the shared-across-batch case, or
    # 3-D (batch, seq, d) for the per-item RoPE case.
    if freqs.ndim == 2:
        freqs = freqs[-seq_len:]
    else:
        freqs = freqs[:, -seq_len:]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class RunningInstanceNorm(nn.Module):
    def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1,1,dim))
        self.register_buffer("running_std", torch.ones(1,1,dim))
        self.saturate = saturate
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.trainable_gain = trainable_gain
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(1))
    
    def _update_stats(self, x):
        self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
        self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        x = (x - self.running_mean) / self.running_std
        if self.saturate:
            x = torch.asinh(x)
        if self.trainable_gain:
            x = x * self.gain
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    #@compile
    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads

        dim_kv = dim_context if dim_context is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            # Cross-attention: use separate projections
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        if causal:
            print('Using `causal` argument disables FlexAttention. If you want to use them together, incorporate causal masking into `flex_attention_block_mask`.')

        # Per-module KV cache (avoids dict mutation inside compiled code)
        self._cache_k = None
        self._cache_v = None

    def set_cache(self, k, v):
        self._cache_k = k
        self._cache_v = v

    def clear_cache(self):
        self._cache_k = None
        self._cache_v = None

    def has_cache(self):
        return self._cache_k is not None

    def _split_qkv_projections_for_cache(self):
        """
        Split fused to_qkv projection into separate to_q and to_kv for KV caching.
        This allows computing only Q for the encoder portion and only K,V for the decoder.
        """
        if hasattr(self, 'to_q') and hasattr(self, 'to_kv'):
            # Already split
            return

        if not hasattr(self, 'to_qkv'):
            raise RuntimeError("Cannot split projections: no to_qkv found")

        # Extract weights from fused projection
        # to_qkv projects to [q, k, v] each of size dim
        if type(self.to_qkv) != nn.Linear:
            raise RuntimeError(f"Unsupported to_qkv type for splitting: {type(self.to_qkv)}")

        qkv_weight = self.to_qkv.weight.data  # Shape: [dim * 3, dim]
        dim = self.dim

        # Split into q, k, v weights
        q_weight = qkv_weight[:dim, :]      # First dim rows
        kv_weight = qkv_weight[dim:, :]     # Remaining 2*dim rows (k and v)

        # Create new linear layers
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        # Copy weights
        self.to_q.weight.data = q_weight
        self.to_kv.weight.data = kv_weight

        # Move to same device as original
        self.to_q = self.to_q.to(self.to_qkv.weight.device)
        self.to_kv = self.to_kv.to(self.to_qkv.weight.device)

        # Delete the fused projection to save memory
        del self.to_qkv

    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k


    def apply_attn(self, q, k, v, causal = None, flex_attention_block_mask = None, flex_attention_score_mod = None, flash_attn_sliding_window = None):

        if self.num_heads != self.kv_heads:
             # Repeat interleave kv_heads to match q_heads for grouped query attention
             heads_per_kv_head = self.num_heads // self.kv_heads
             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        flash_attn_available = flash_attn_func is not None
        if flash_attn_sliding_window is not None and (not flash_attn_available):
            print(f"Cannot use FlashAttention sliding window as FlashAttention is disabled or not available")

        if (flex_attention_block_mask is not None or flex_attention_score_mod is not None) and flash_attn_sliding_window is not None:
            print(f"cannot use both FlashAttention and FlexAttention, favouring FlexAttention")

        if causal and (flex_attention_block_mask is not None or flex_attention_score_mod is not None):
            print(f"Disabling FlexAttention because causal is set")
            flex_attention_block_mask = None
            flex_attention_score_mod = None

        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            out = flex_attention_compiled(q,k,v,
                block_mask = flex_attention_block_mask,
                score_mod = flex_attention_score_mod)        
        elif flash_attn_available:
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))
            
            out = flash_attn_func(q, k, v, causal = causal, window_size=flash_attn_sliding_window if (flash_attn_sliding_window is not None) else [-1,-1])
            
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal = causal)
        return out


    #@compile
    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        causal = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        use_kv_cache = False,
        cache_initialized = False,
        encoder_seq_len = None,
        update_cache_slice = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        using_cache = use_kv_cache
        is_cross_attn = has_context

        # Split fused projection if using cache for first time
        if using_cache and not cache_initialized and not is_cross_attn:
            if not hasattr(self, 'to_q') and hasattr(self, 'to_qkv'):
                self._split_qkv_projections_for_cache()

        # Compute Q,K,V with caching support
        if using_cache and cache_initialized:
            # Cache is initialized - reuse cached K,V where possible
            if is_cross_attn:
                # Cross-attention: reuse entire cached K,V (text conditioning is constant)
                k = self._cache_k
                v = self._cache_v
                # Always compute Q (needed for every forward pass)
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
            else:
                # Self-attention: reuse encoder K,V, compute decoder K,V
                # Note: x is already sliced to decoder-only when cache is initialized
                k_encoder = self._cache_k
                v_encoder = self._cache_v

                # Compute Q and K,V for decoder (x is already decoder-only)
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)

                k_decoder, v_decoder = self.to_kv(x).chunk(2, dim=-1)
                k_decoder, v_decoder = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k_decoder, v_decoder))

                # Concatenate encoder + decoder
                k = torch.cat([k_encoder, k_decoder], dim=2)
                v = torch.cat([v_encoder, v_decoder], dim=2)
        else:
            # Normal path: compute full K,V (no cache or first pass)
            if hasattr(self, 'to_q'):
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        # Cached K,V already have normalization baked in from the first pass,
        # so we must avoid re-normalizing them.
        if using_cache and cache_initialized:
            if is_cross_attn:
                # Cross-attention: entire K is from cache (already normalized).
                # Only normalize Q.
                if self.qk_norm == "l2":
                    q = F.normalize(q, dim=-1)
                elif self.qk_norm != "none":
                    q = self.q_norm(q).to(q.dtype)
            else:
                # Self-attention: encoder K is from cache (already normalized),
                # decoder K is fresh and needs normalization.
                if self.qk_norm == "l2":
                    q = F.normalize(q, dim=-1)
                    k_decoder = k[:, :, encoder_seq_len:]
                    k_decoder = F.normalize(k_decoder, dim=-1)
                    k = torch.cat([k[:, :, :encoder_seq_len], k_decoder], dim=2)
                elif self.qk_norm != "none":
                    k_decoder = k[:, :, encoder_seq_len:]
                    q, k_decoder = self.apply_qk_layernorm(q, k_decoder)
                    k = torch.cat([k[:, :, :encoder_seq_len], k_decoder], dim=2)
        else:
            if self.qk_norm == "l2":
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)
            elif self.qk_norm != "none":
                q, k = self.apply_qk_layernorm(q, k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            if using_cache and cache_initialized:
                ratio = 1
                q_freqs, k_freqs = freqs, freqs
            elif q.shape[-2] >= k.shape[-2]:
                ratio = q.shape[-2] / k.shape[-2]
                q_freqs, k_freqs = freqs, ratio * freqs
            else:
                ratio = k.shape[-2] / q.shape[-2]
                q_freqs, k_freqs = ratio * freqs, freqs

            # Special handling for cached self-attention with encoder/decoder split
            if using_cache and cache_initialized and not is_cross_attn:
                cached_encoder_seq_len = encoder_seq_len

                # Slice freqs along the seq dimension (dim 0 for 2D, dim 1 for 3D batch freqs)
                if q_freqs.ndim == 2:
                    q_freqs_decoder = q_freqs[cached_encoder_seq_len:]
                    k_freqs_decoder = k_freqs[cached_encoder_seq_len:]
                else:
                    q_freqs_decoder = q_freqs[:, cached_encoder_seq_len:]
                    k_freqs_decoder = k_freqs[:, cached_encoder_seq_len:]

                # Q is decoder-only and needs RoPE with position offset
                q = apply_rotary_pos_emb(q, q_freqs_decoder)

                # K encoder portion already has RoPE applied (from cache)
                # Only apply RoPE to decoder portion with position offset
                k_decoder = k[:, :, cached_encoder_seq_len:]
                k_decoder = apply_rotary_pos_emb(k_decoder, k_freqs_decoder)

                # Concatenate: [cached_encoder_k (already has RoPE), decoder_k (just applied RoPE)]
                k = torch.cat([k[:, :, :cached_encoder_seq_len], k_decoder], dim=2)
            else:
                # Normal path: apply RoPE to full sequences
                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        # Populate cache on first pass (after RoPE has been applied)
        if using_cache and not self.has_cache():
            # with torch.no_grad():
            if is_cross_attn:
                # Cache entire cross-attention K,V
                self.set_cache(k.detach().clone().contiguous(), v.detach().clone().contiguous())
            else:
                # Cache encoder portion of self-attention K,V
                if encoder_seq_len is not None and encoder_seq_len > 0:
                    self.set_cache(k[:, :, :encoder_seq_len].detach().clone().contiguous(), v[:, :, :encoder_seq_len].detach().clone().contiguous())

        # Atomically eject+extend the cache (for block-causal streaming first step).
        # Slice (start, end) selects the new cache contents from the full K,V: this simultaneously
        # drops the oldest chunk (start>0) and incorporates the freshly computed pending chunk.
        # Keeping cache shape constant across steps avoids tensor-shape oscillation inside
        # torch.compile'd TransformerBlock.forward (prior cause of illegal memory access).
        if using_cache and update_cache_slice is not None and not is_cross_attn:
            if isinstance(update_cache_slice[0], (tuple, list)):
                # Multi-piece slice: assemble new cache from non-contiguous K,V ranges.
                # Used when accomp cutoff invalidates a strict-interior subset of cached chunks
                # (so the surviving-cached prefix and the freshly-recomputed chunks are not
                # contiguous in the K tensor).
                new_k = torch.cat([k[:, :, s:e] for s, e in update_cache_slice], dim=2)
                new_v = torch.cat([v[:, :, s:e] for s, e in update_cache_slice], dim=2)
                self.set_cache(new_k.detach().clone().contiguous(), new_v.detach().clone().contiguous())
            else:
                start, end = update_cache_slice
                self.set_cache(
                    k[:, :, start:end].detach().clone().contiguous(),
                    v[:, :, start:end].detach().clone().contiguous()
                )

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            out_dc = out.mean(dim=-2, keepdim=True)
            out_hf = out - out_dc

            # Selectively modulate DC and high frequency components
            out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    #@compile
    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            add_rope = False,
            layer_scale = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads,dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
       
        if layer_scale and zero_init_branch_outputs:
            print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
            zero_init_branch_outputs = False
            
        self.pre_norm = LayerNorm(dim,**norm_kwargs) if not remove_norms else DynamicTanh(dim)

        self.add_rope = add_rope

        self.self_attn = Attention(
            dim,
            dim_heads = self.dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
            self.cross_attn = Attention(
                dim,
                dim_heads = self.dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.layer_ix = layer_ix

        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None
        
    @compile
    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        rotary_pos_emb = None,
        self_attention_block_mask = None,
        self_attention_score_mod = None,
        cross_attention_block_mask = None,
        cross_attention_score_mod = None,
        self_attention_flash_sliding_window = None,
        cross_attention_flash_sliding_window = None,
        use_kv_cache = False,
        cache_initialized = False,
        encoder_seq_len = None,
        update_cache_slice = None,
        cross_attn_no_cache = False,
    ):
        # When time-varying cross-attn conditioning is used, the cached cross-attn K,V
        # become stale as the prompt evolves. This flag forces cross-attn to always
        # recompute K,V from `context` and never populate its module cache.
        cross_use_kv_cache = use_kv_cache and not cross_attn_no_cache
        cross_cache_initialized = cache_initialized and not cross_attn_no_cache
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:

            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, use_kv_cache = use_kv_cache, cache_initialized = cache_initialized, encoder_seq_len = encoder_seq_len, update_cache_slice = update_cache_slice)
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window, use_kv_cache = cross_use_kv_cache, cache_initialized = cross_cache_initialized))

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual

        else:
            x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, use_kv_cache = use_kv_cache, cache_initialized = cache_initialized, encoder_seq_len = encoder_seq_len, update_cache_slice = update_cache_slice))

            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window, use_kv_cache = cross_use_kv_cache, cache_initialized = cross_cache_initialized))

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = x + self.ff_scale(self.ff(self.ff_norm(x)))
        return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None
            print('Rotary positional embeddings are disabled.')

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def clear_kv_cache(self):
        """Clear KV caches stored on all attention modules."""
        for layer in self.layers:
            layer.self_attn.clear_cache()
            if hasattr(layer, 'cross_attn') and layer.cross_attn is not None:
                layer.cross_attn.clear_cache()

    def forward(
        self,
        x,
        prepend_embeds = None,
        global_cond = None,
        return_info = False,
        use_checkpointing = False,
        exit_layer_ix = None,
        input_add_emb = None,
        context_router_mask = None,
        use_kv_cache = False,
        kv_cache = None,
        postpend=None,
        rotary_seq_len = None,
        per_item_rope_positions = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        use_checkpointing = os.environ.get('USE_CHECKPOINTING', '1') == '1' or use_checkpointing

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if context_router_mask is not None:
            # reshape context_router_mask to batch seq channels
            context_router_mask = context_router_mask.transpose(-1, -2)
            x = x * context_router_mask

        if input_add_emb is not None:
            x = x + input_add_emb

        prepend_length = 0
        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            if not postpend:
                x = torch.cat((prepend_embeds, x), dim = -2)
            else:
                x = torch.cat((x, prepend_embeds), dim = -2)

        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)

        if self.rotary_pos_emb is not None:
            if per_item_rope_positions is not None:
                # per_item_rope_positions: (batch, original_seq_len) — positions for the main sequence
                # tokens, not including the prepend/postpend token(s). We extend them here to account
                # for the prepend/postpend token, which always gets rope position 0.
                device = per_item_rope_positions.device
                b_rope = per_item_rope_positions.shape[0]
                if prepend_length > 0:
                    # Shift all sequence positions up by prepend_length so position 0 is reserved
                    # for the prepend/postpend token(s).
                    seq_positions = per_item_rope_positions + prepend_length  # (batch, seq_len)
                    prepend_pos = torch.zeros(b_rope, prepend_length, dtype=torch.long, device=device)
                    if not postpend:
                        # Prepend: [0(x prepend_length), shifted_seq_positions]
                        full_positions = torch.cat([prepend_pos, seq_positions], dim=1)
                    else:
                        # Postpend: [shifted_seq_positions, 0(x prepend_length)]
                        # The postpended token(s) explicitly get position 0, equivalent to the
                        # standard roll behaviour but expressed directly as position indices.
                        full_positions = torch.cat([seq_positions, prepend_pos], dim=1)
                else:
                    full_positions = per_item_rope_positions
                rotary_pos_emb = self.rotary_pos_emb.forward_from_positions(full_positions)
                # No roll needed — positions are set explicitly above.
            else:
                rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1] if rotary_seq_len is None else rotary_seq_len)
                if postpend:
                    # If postpending, we need to shift the RoPE frequencies such that the prepend_cond tokens (which are now at the end of the sequence) get the correct RoPE frequencies
                    # since they are expecting to be at the beginning of the sequence. This is done by rolling the RoPE frequencies by the length of the postpended tokens.
                    rolled_freqs_0 = torch.roll(rotary_pos_emb[0], shifts=-1, dims=0) # TODO
                    rotary_pos_emb = (rolled_freqs_0, rotary_pos_emb[1])

        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Extract encoder sequence length for KV caching
        encoder_seq_len = None
        if use_kv_cache and kv_cache is not None:
            if 'encoder_seq_len' in kv_cache:
                encoder_seq_len = kv_cache['encoder_seq_len']
            elif context_router_mask is not None:
                # Context (encoder) tokens are the non-target positions. context_router_mask = 1 - inpaint_mask,
                # so the context region is exactly where the mask is 0.
                mask_row = context_router_mask.reshape(-1, context_router_mask.shape[-1])[0]
                encoder_seq_len = int((mask_row == 0).sum().item())

        # Initialize KV cache structure on first use
        cache_initialized = False
        if use_kv_cache and kv_cache is not None:
            cache_initialized = kv_cache.get('initialized', False)
            if not kv_cache.get('initialized', False):
                if 'encoder_seq_len' not in kv_cache and encoder_seq_len is not None:
                    kv_cache['encoder_seq_len'] = encoder_seq_len
            
        # Disable checkpointing when update_cache_slice is set (block-causal fused first step).
        # Gradient checkpointing recomputes each layer during backward, but set_cache mutations
        # to self._cache_k/v (module attributes, not saved inputs) persist between forward and
        # recomputation, causing the recomputed forward to read the already-expanded cache and
        # produce mismatched tensor dimensions (CUDA illegal memory access).
        skip_checkpointing = kwargs.get('update_cache_slice', None) is not None

        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):

            if use_checkpointing and not skip_checkpointing:
                x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, self_attention_flash_sliding_window = self.sliding_window, use_kv_cache = use_kv_cache, cache_initialized = cache_initialized, encoder_seq_len = encoder_seq_len, **kwargs)
            else:
                x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, self_attention_flash_sliding_window = self.sliding_window, use_kv_cache = use_kv_cache, cache_initialized = cache_initialized, encoder_seq_len = encoder_seq_len, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x
