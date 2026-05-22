import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask

from .blocks import FourierFeatures
from .transformer import ContinuousTransformer, apply_rotary_pos_emb

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        input_add_dims=[],
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "v_denoiser", "rectified_flow", "rf_denoiser"] = "v",
        postpend=False,
        split_qkv=False,
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if len(input_add_dims) > 0:
            # Input add conditioning, module dict
            # self.to_input_add_embed = nn.ModuleDict()
            # for id, dim in input_add_dims.items():
            #     self.to_input_add_embed[id] = nn.Linear(dim, embed_dim, bias=False)
            # self.input_add_cond_cache = None
            # convert to just a single concatenated linear layer
            # input_add_dims is now an ordered list of tuples (id, dim)
            self.input_add_dims = input_add_dims
            total_input_add_dim = sum([dim for id, dim in input_add_dims])
            self.to_input_add_embed = nn.Linear(total_input_add_dim, embed_dim, bias=False)

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size
        self.postpend = postpend

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        if split_qkv:
            for block in self.transformer.layers:
                block.self_attn._split_qkv_projections_for_cache()

    def init_kv_cache(self, batch_size, encoder_seq_len, device, dtype, context_seq_len=None):
        """Pre-allocate KV cache buffers. Call once before the sampling loop for reduce-overhead compatibility."""
        self.transformer.init_kv_cache(batch_size, encoder_seq_len, device, dtype, context_seq_len=context_seq_len)

    def clear_kv_cache(self):
        """Zero out KV cache buffers and reset initialized flags (keeps allocations alive)."""
        self.transformer.clear_kv_cache()

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        context_router_mask=None,
        use_kv_cache=False,
        kv_cache=None,
        postpend=None,
        prefill=False,
        block_causal_first_step=False,
        update_cache_slice=None,
        decoder_start=None,
        **kwargs):

        postpend = self.postpend if postpend is None else postpend

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        add_emb = self.to_input_add_embed(input_add_cond.transpose(1, 2)) if input_add_cond is not None else None
        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        if context_router_mask is not None:
            x = x * context_router_mask

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        # Prefill: if KV cache exists but not yet initialized, run encoder-only pass to populate it.
        # This lets all N denoising steps use fast decoder-only flash attention instead of flex attention.
        if use_kv_cache and kv_cache is not None and not kv_cache.get('initialized', False) and prefill:
            enc_seq_len_prefill = 208
            kv_cache['encoder_seq_len'] = enc_seq_len_prefill
            if enc_seq_len_prefill is not None and enc_seq_len_prefill > 0:
                x_enc = x[:, :enc_seq_len_prefill]
                add_emb_enc = add_emb[:, :enc_seq_len_prefill] if add_emb is not None else None
                # Exclude flex-attention block mask — use standard flash attention for prefill
                if kwargs.get('self_attention_block_mask', None) is not None:
                    kwargs.pop('self_attention_block_mask') # this should turn off flex attention
                enc_out = self.transformer(
                    x_enc,
                    # prepend_embeds=prepend_inputs,
                    context=cross_attn_cond,
                    return_info=False,
                    input_add_emb=add_emb_enc,
                    context_router_mask=None,
                    use_kv_cache=True,
                    kv_cache=kv_cache,
                    postpend=postpend,
                    **extra_args,
                    **kwargs,
                )
                # Run encoder output through the same post-processing pipeline, then cache it
                enc_out = rearrange(enc_out, "b t c -> b c t")
                # if not postpend:
                #     enc_out = enc_out[:, :, prepend_length:]
                # else:
                #     enc_out = enc_out[:, :, :-prepend_length] if prepend_length > 0 else rearrange(enc_out, "b t c -> b c t")
                if self.patch_size > 1:
                    enc_out = rearrange(enc_out, "b (c p) t -> b c (t p)", p=self.patch_size)
                enc_out = self.postprocess_conv(enc_out) + enc_out
                kv_cache['encoder_output'] = enc_out.detach()
                kv_cache['initialized'] = True

        # When KV cache is initialized (by prefill above or previous call), only pass decoder portion through the network
        cache_is_initialized = use_kv_cache and kv_cache is not None and kv_cache.get('initialized', False)
        uses_block_causal_stream = block_causal_first_step
        if cache_is_initialized:
            encoder_seq_len = kv_cache['encoder_seq_len']

            # Slice x to decoder only. On the block-causal fused first step, decoder_start
            # < encoder_seq_len by one chunk: the per-layer cache size is nwc*cs while x's
            # encoder portion only covers (nwc-1)*cs tokens (the oldest chunk still sits in
            # the cache and will be atomically ejected via update_cache_slice).
            x_decoder_start = decoder_start if (uses_block_causal_stream and decoder_start is not None) else encoder_seq_len
            rotary_seq_len = x.shape[1] + prepend_length if prepend_inputs is not None else x.shape[1]
            kwargs['rotary_seq_len'] = rotary_seq_len
            x = x[:, x_decoder_start:]

            # Truncate input_add_emb to decoder portion (should be zeros there anyway)
            if add_emb is not None:
                add_emb = add_emb[:, x_decoder_start:]

            if uses_block_causal_stream:
                # Block-causal streaming (fused-first or per-step): keep context_router_mask and flex attention mask.
                # Slice context_router_mask to match the decoder portion.
                if context_router_mask is not None:
                    context_router_mask = context_router_mask[:, :, x_decoder_start:]
                # self_attention_block_mask is kept as-is (caller provides the right mask)
            else:
                # Standard enc-dec cache path: bidirectional attention, no mask
                context_router_mask = None
                if kwargs.get('self_attention_block_mask', None) is not None:
                    kwargs.pop('self_attention_block_mask')

        # Thread update_cache_slice to the transformer for any block-causal streaming call.
        # For per-step bootstrap (cache not yet initialized), this makes the very first forward
        # populate the per-layer cache with the sliding-window slice K[start:end] = last nwc chunks,
        # rather than the default "first encoder_seq_len tokens" (which would keep the oldest chunk).
        if uses_block_causal_stream and update_cache_slice is not None:
            kwargs['update_cache_slice'] = update_cache_slice

        if self.transformer_type == "continuous_transformer":
            # Masks not currently implemented for continuous transformer
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, return_info=return_info, exit_layer_ix=exit_layer_ix, input_add_emb=add_emb, context_router_mask=context_router_mask, use_kv_cache=use_kv_cache, kv_cache=kv_cache, postpend=postpend, **extra_args, **kwargs)

            if return_info:
                output, info = output

            # Avoid postprocessing on early exit
            if exit_layer_ix is not None:
                if return_info:
                    return output, info
                else:
                    return output



        if not postpend:
            output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]
        else:
            output = rearrange(output, "b t c -> b c t")[:,:,:-prepend_length] if prepend_length > 0 else rearrange(output, "b t c -> b c t")

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        # Cache encoder output on first pass, or restore it on subsequent passes
        if use_kv_cache and kv_cache is not None:
            if not kv_cache.get('initialized', False):
                # First pass: cache encoder portion of output
                encoder_seq_len = kv_cache.get('encoder_seq_len')
                if encoder_seq_len is not None and encoder_seq_len > 0:
                    kv_cache['encoder_output'] = output[..., :encoder_seq_len].detach()
                kv_cache['initialized'] = True
            elif 'encoder_output' in kv_cache:
                # Subsequent passes: prepend cached encoder output to decoder output
                kv_cache['initialized'] = True
                encoder_output = kv_cache['encoder_output']
                output = torch.cat([encoder_output, output], dim=-1)

                if block_causal_first_step and update_cache_slice is not None:
                    # Fused first step: atomically eject oldest chunk and add pending to
                    # encoder_output. cat produced (nwc+2)*cs tokens ([old_cache|pending|target]);
                    # slice [start:end] writes the new encoder cache; slice [start:] keeps the
                    # caller-facing output at sample_size = (nwc+1)*cs.
                    if isinstance(update_cache_slice[0], (tuple, list)):
                        # Multi-piece slice for accomp-cutoff < context_size: surviving-cached
                        # prefix and freshly-recomputed chunks live in non-contiguous K positions.
                        # Cache = concat of all pieces; caller-facing output = first piece +
                        # everything from the second piece onward (which already includes the
                        # target chunk that sits past the last cache piece's end).
                        cache_pieces = [output[..., s:e] for s, e in update_cache_slice]
                        kv_cache['encoder_output'] = torch.cat(cache_pieces, dim=-1).detach()
                        first_start, first_end = update_cache_slice[0]
                        last_start, _ = update_cache_slice[-1]
                        output = torch.cat([
                            output[..., first_start:first_end],
                            output[..., last_start:],
                        ], dim=-1)
                    else:
                        start, end = update_cache_slice
                        kv_cache['encoder_output'] = output[..., start:end].detach()
                        output = output[..., start:]

        if return_info:
            return output, info
        return output

    def apg_project(self, v0, v1):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_norm_threshold=0.0,
        cfg_interval = (0, 1),
        scale_phi=0.0,
        plus_plus=False,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        context_router_mask=None,
        use_kv_cache=False,
        kv_cache=None,
        **kwargs):


        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        if input_add_cond is not None:
            # Convert input_add_cond to the model dtype
            input_add_cond = input_add_cond.to(model_dtype)

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                input_add_cond=input_add_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                context_router_mask=context_router_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:

            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)


            if input_add_cond is not None:
                # get dims, apply dropout to each individual conditioning
                # input_add_cond is a concat of multiple conditionings into 1 tensor along channel dim
                # self.input_add_dims is ordered list of tuples (id, dim)
                total_dim = input_add_cond.shape[1]
                start_idx = 0
                for id, dim in self.input_add_dims:
                    end_idx = start_idx + dim
                    null_embed = torch.zeros_like(input_add_cond[:, start_idx:end_idx, :], device=input_add_cond.device)
                    dropout_mask = torch.bernoulli(torch.full((input_add_cond.shape[0], 1, 1), cfg_dropout_prob, device=input_add_cond.device)).to(torch.bool)
                    input_add_cond[:, start_idx:end_idx, :] = torch.where(dropout_mask, null_embed, input_add_cond[:, start_idx:end_idx, :])
                    start_idx = end_idx


        if self.diffusion_objective in ["v", "v_denoiser"]:
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            if input_add_cond is not None:
                batch_input_add_cond = torch.cat([input_add_cond, input_add_cond], dim=0)
            else:
                batch_input_add_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            if context_router_mask is not None:
                batch_context_router_mask = torch.cat([context_router_mask, context_router_mask], dim=0)
            else:
                batch_context_router_mask = None

            # Double per_item_rope_positions if present (batch is doubled for CFG)
            if 'per_item_rope_positions' in kwargs:
                p = kwargs['per_item_rope_positions']
                kwargs = {**kwargs, 'per_item_rope_positions': torch.cat([p, p], dim=0)}

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask = batch_masks,
                input_concat_cond=batch_input_concat_cond,
                input_add_cond=batch_input_add_cond,
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                context_router_mask = batch_context_router_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            # CFG++ path: return both branches; caller handles the per-sampler update.
            if plus_plus:
                if return_info:
                    info["uncond_output"] = uncond_output
                    return (cond_output, uncond_output), info
                return cond_output, uncond_output

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale
                
            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
                
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                input_add_cond=input_add_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                context_router_mask=context_router_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs
            )


class BlockCausalStreamingCache:
    """Manages KV cache state for block-causal sliding window streaming generation.

    Lifecycle:
        1. Bootstrap: run full [context | target] through model with block-causal mask.
           This populates attention KV caches for context and kv_cache['encoder_output'].
        2. Denoise first chunk steps 1..N: standard cached path (bidirectional).
        3. advance(clean_chunk): eject oldest chunk from cache, set pending.
        4. Fused first step: pass [pending | target] with block-causal mask,
           expand cache to include pending.
        5. Denoise steps 1..N: standard cached path.
        6. Repeat from 3.

    Usage:
        cache = BlockCausalStreamingCache(dit, chunk_size=48, n_window_chunks=4)

        # Bootstrap (first chunk)
        kv_cache = cache.init_kv_cache()
        for step in range(n_steps):
            if step == 0:
                kwargs = {'self_attention_block_mask': training_mask, 'use_kv_cache': True, 'kv_cache': kv_cache}
            else:
                kwargs = {'use_kv_cache': True, 'kv_cache': kv_cache}
            output = model(x, t, cond=cond, context_router_mask=context_router_mask, **kwargs)
        cache.advance(clean_chunk)

        # Subsequent chunks
        for chunk_idx in range(1, n_chunks):
            for step in range(n_steps):
                if step == 0:
                    kwargs = cache.get_first_step_kwargs(device)
                    kwargs.update({'use_kv_cache': True, 'kv_cache': kv_cache})
                    output = model(x, t, cond=cond, context_router_mask=context_router_mask, **kwargs)
                    cache.finalize_first_step()
                else:
                    output = model(x, t, cond=cond, use_kv_cache=True, kv_cache=kv_cache)
            cache.advance(clean_chunk)
    """

    def __init__(self, dit, chunk_size, n_window_chunks, rope_mode=None, accomp_cutoff=None):
        """
        Args:
            dit: The DiffusionCondDITWrapper (model.model)
            chunk_size: Size of each chunk in latent frames
            n_window_chunks: Number of context chunks in the sliding window
            rope_mode: "offset" (positions grow), "fixed" (always [0, seq_len]), or None (no RoPE)
            accomp_cutoff: Accompaniment cutoff in latent frames (0..context_size). When set and
                less than context_size, the chunk straddling the cutoff boundary newly enters the
                "accomp present" zone after each window slide, so its cached K,V (computed under
                the previous window where its accomp position was past cutoff = zero) is stale.
                We extend the fused first step's decoder to recompute chunks
                [floor(cutoff/cs)-1, n_window_chunks - 1) along with the standard pending chunk.
        """
        self.dit = dit
        self.chunk_size = chunk_size
        self.n_window_chunks = n_window_chunks
        self.rope_mode = rope_mode
        self.accomp_cutoff = accomp_cutoff
        self.position_offset = 0  # global chunk index of window start
        self.pending_chunk = None
        self.kv_cache = None
        self._first_step_mask = None

    @property
    def j_boundary(self):
        """Lowest c+1 chunk index that needs cache recomputation in the fused first step.

        Returns n_window_chunks - 1 (= standard fused first step, only pending recomputed) when
        accomp_cutoff is unset or covers the full context.
        """
        nwc = self.n_window_chunks
        cs = self.chunk_size
        if self.accomp_cutoff is None or self.accomp_cutoff >= nwc * cs:
            return nwc - 1
        # User spec: chunks [floor(cutoff/cs)-1, nwc-1) need recompute. j_boundary = lower bound.
        return max(0, self.accomp_cutoff // cs - 1)

    @property
    def context_len(self):
        """Total context length (all cached chunks, not including pending or target)."""
        return self.n_window_chunks * self.chunk_size

    @property
    def seq_len(self):
        """Full sequence length (context + target)."""
        return (self.n_window_chunks + 1) * self.chunk_size

    def init_kv_cache(self):
        """Create a fresh kv_cache dict and clear attention caches. Call before bootstrap."""
        self.dit.model.transformer.clear_kv_cache()
        self.kv_cache = {
            'initialized': False,
            'encoder_seq_len': self.context_len,
        }
        self.pending_chunk = None
        self.position_offset = 0
        return self.kv_cache

    def advance(self, clean_chunk):
        """After generating a chunk, record pending. DOES NOT touch per-layer caches.

        The ejection of the oldest chunk and the addition of the new pending chunk happen
        atomically during the fused first step (see BlockCausalStreamingCache docstring).
        Keeping the per-layer cache shape constant across steps is critical: previously we
        shrunk the cache here and re-grew it on the fused step, which caused tensor-shape
        oscillation on module attributes inside torch.compile'd TransformerBlock.forward
        and produced CUDA illegal memory access errors.

        Args:
            clean_chunk: (B, C, chunk_size) the fully denoised chunk
        """
        self.pending_chunk = clean_chunk.detach()
        self.position_offset += 1

    def get_first_step_mask(self, device):
        """Get the flex attention block mask for the fused first step.

        Standard (j_boundary == nwc-1): Q = [pending | target | postpend], KV = [cached_full |
        pending | target | postpend]. Mask enforces block-causal attention.

        Extended (j_boundary < nwc-1, when accomp cutoff < context): Q extends to chunks
        [j_boundary, nwc] of c+1 (so to_kv recomputes them with current-window accomp). The KV
        still includes the full prior cache (length nwc*cs), but the cached entries for c+1
        chunks j_boundary..nwc-2 (= c chunks j_boundary+1..nwc-1, occupying cached positions
        (j_boundary+1)*cs..nwc*cs) are stale duplicates of the freshly-computed versions and
        get masked out.
        """
        if self._first_step_mask is not None:
            return self._first_step_mask

        cs = self.chunk_size
        nwc = self.n_window_chunks
        jb = self.j_boundary
        # Indexing convention (matches existing code): chunk 0 = oldest cached (about to eject),
        # chunks 1..nwc-1 = surviving cached, chunk nwc = pending, chunk nwc+1 = target.
        n_total_chunks = nwc + 2
        n_cached_tokens = nwc * cs

        # Q covers c+1 chunks [j_boundary, nwc] + postpend → fresh-portion length + 1.
        fresh_len = (nwc + 1 - jb) * cs
        q_len = fresh_len + 1
        kv_len = n_cached_tokens + fresh_len + 1
        # Q's first token's chunk in the n_total_chunks indexing: j_boundary maps to jb+1.
        q_chunk_offset_in_total = jb + 1

        # Stale cached range (in cached-portion KV positions): c chunks (jb+1)..(nwc-1) =
        # cached positions (jb+1)*cs..nwc*cs. Empty when jb == nwc-1 (standard case).
        stale_cached_start = (jb + 1) * cs
        stale_cached_end = nwc * cs

        def first_step_mask_fn(b, h, q_idx, kv_idx):
            # Q chunk in n_total_chunks indexing.
            q_chunk = (q_idx // cs + q_chunk_offset_in_total).clamp(max=n_total_chunks - 1)
            kv_chunk = (kv_idx // cs).clamp(max=n_total_chunks - 1)
            block_causal = (kv_chunk <= q_chunk) & (kv_chunk >= q_chunk - nwc + 1)
            # Mask out stale cached duplicates (only relevant when extended, i.e., jb < nwc-1).
            is_stale = (kv_idx >= stale_cached_start) & (kv_idx < stale_cached_end)
            return block_causal & ~is_stale

        self._first_step_mask = create_block_mask(
            first_step_mask_fn, B=None, H=None,
            Q_LEN=q_len, KV_LEN=kv_len,
            device=device, _compile=False  # _compile=False avoids CUDA illegal memory access
        )
        return self._first_step_mask

    def get_first_step_kwargs(self, device):
        """Get extra kwargs to pass to the model for the fused first step.

        Returns a dict with block_causal_first_step, update_cache_slice (atomic
        eject+extend), decoder_start (x-slicing offset for decoder portion), and
        self_attention_block_mask. Merge these into your model call kwargs.
        """
        assert self.pending_chunk is not None, "No pending chunk — call advance() first"
        cs = self.chunk_size
        nwc = self.n_window_chunks
        jb = self.j_boundary
        if jb == nwc - 1:
            # Standard: drop oldest, keep surviving + pending. Single contiguous slice.
            update_slice = (cs, cs + nwc * cs)
        else:
            # Extended (accomp cutoff < context): cache = surviving prefix [cs..(jb+1)*cs] (c+1
            # chunks 0..jb-1) + freshly-recomputed chunks [nwc*cs..(2*nwc-jb)*cs] (c+1 chunks
            # jb..nwc-1). Two non-contiguous K-tensor slices, total length nwc*cs (unchanged).
            update_slice = ((cs, (jb + 1) * cs), (nwc * cs, (2 * nwc - jb) * cs))
        # Decoder covers c+1 chunks [j_boundary..nwc] of x; standard reduces to chunks [nwc-1, nwc].
        dec_start = jb * cs
        return {
            'block_causal_first_step': True,
            'update_cache_slice': update_slice,
            'decoder_start': dec_start,
            'self_attention_block_mask': self.get_first_step_mask(device),
        }

    def finalize_first_step(self):
        """Call after the fused first step completes.

        The per-layer caches have already been atomically updated via update_cache_slice.
        In "fixed" RoPE mode we now re-rotate the just-written cache to shift its baked
        positions back by chunk_size (so they line up with the next step's fixed positions
        [0..nwc*cs-1] instead of [cs..cs+nwc*cs-1]). In "offset" mode no rotation needed —
        cached K retains its absolute-frame positions.
        """
        if self.rope_mode == "fixed":
            self._rerotate_cached_keys()
        self.pending_chunk = None

    def get_rope_positions(self, batch_size, device, first_step=False):
        """Get per_item_rope_positions for the current window.

        Returns (batch, L) integer positions for the main sequence tokens
        (excluding postpend — that's handled by ContinuousTransformer).

        When first_step=True (block-causal fused first step on chunk_idx > 0), L = (nwc+2)*cs
        (the fused KV spans one extra chunk — the about-to-be-ejected oldest chunk still in
        the cache). The window effectively starts one chunk earlier than the current offset.

        When first_step=False (bootstrap / steps 1..N), L = (nwc+1)*cs = seq_len.

        In "offset" mode positions grow with position_offset. In "fixed" mode they always
        start at 0 (cached K is re-rotated in finalize_first_step to stay consistent).
        Returns None if rope_mode is None.
        """
        if self.rope_mode is None:
            return None

        if first_step:
            cs = self.chunk_size
            nwc = self.n_window_chunks
            jb = self.j_boundary
            cached_len = nwc * cs
            fresh_len = (nwc + 1 - jb) * cs
            # Fresh freqs are placed at the +cs shifted positions [(jb+1)*cs, (nwc+2)*cs); after
            # the post-fact _rerotate_cached_keys(-cs) shift, the new cache lands at the c+1
            # fixed positions [0, nwc*cs). For jb==nwc-1 (standard) this collapses to the
            # contiguous range [nwc*cs, (nwc+2)*cs), matching the existing implementation.
            offset_start = max(self.position_offset - 1, 0) * cs
            if self.rope_mode == "fixed":
                cached_part = torch.arange(cached_len, device=device)
                fresh_part = torch.arange((jb + 1) * cs, (nwc + 2) * cs, device=device)
            elif self.rope_mode == "offset":
                cached_part = torch.arange(offset_start, offset_start + cached_len, device=device)
                fresh_part = torch.arange(offset_start + (jb + 1) * cs, offset_start + (nwc + 2) * cs, device=device)
            else:
                raise ValueError(f"Unknown rope_mode: {self.rope_mode}")
            positions = torch.cat([cached_part, fresh_part])
            return positions.unsqueeze(0).expand(batch_size, -1).long()

        length = self.seq_len
        offset_start = self.position_offset * self.chunk_size

        if self.rope_mode == "fixed":
            positions = torch.arange(length, device=device)
        elif self.rope_mode == "offset":
            positions = torch.arange(offset_start, offset_start + length, device=device)
        else:
            raise ValueError(f"Unknown rope_mode: {self.rope_mode}")

        return positions.unsqueeze(0).expand(batch_size, -1).long()

    def _rerotate_cached_keys(self):
        """Re-rotate cached K entries to shift RoPE positions back by chunk_size.

        Used in "fixed" mode after ejecting the oldest chunk: all remaining
        cached tokens shift one chunk_size earlier in position space.

        Uses the approach from Section 2.3 of Block-Attention (Ma et al., 2025):
        rotate by delta = -chunk_size to shift all positions back.
        """
        transformer = self.dit.model.transformer
        rope_emb = transformer.rotary_pos_emb
        if rope_emb is None:
            return

        inv_freq = rope_emb.inv_freq  # (d/2,)
        device = inv_freq.device

        # Compute rotation freqs for delta = -chunk_size
        # The prepend_length shift (+1 for postpend) doesn't affect the delta.
        delta = torch.tensor([-self.chunk_size], dtype=torch.float32, device=device)
        delta_freqs = torch.einsum('i, j -> i j', delta, inv_freq)  # (1, d/2)
        delta_freqs = torch.cat((delta_freqs, delta_freqs), dim=-1)  # (1, d)

        for layer in transformer.layers:
            attn = layer.self_attn
            if attn.has_cache():
                cached_k = attn._cache_k  # (B, H, cached_len, d)
                # Expand delta_freqs to match cached_len
                cached_len = cached_k.shape[2]
                freqs = delta_freqs.expand(cached_len, -1)  # (cached_len, d)
                # apply_rotary_pos_emb expects (B, H, seq, d) for t and (seq, d) for freqs
                rerotated_k = apply_rotary_pos_emb(cached_k.float(), freqs).to(cached_k.dtype)
                attn.set_cache(rerotated_k.detach().clone().contiguous(), attn._cache_v.detach().clone().contiguous())

