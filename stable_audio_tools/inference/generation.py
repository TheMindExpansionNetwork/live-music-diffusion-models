import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop
from torch.nn.attention.flex_attention import create_block_mask, or_masks
from ..models.dit import BlockCausalStreamingCache
from ..models.conditioners import T5Conditioner, mix_t5_prompts


def _encode_chunk_prompt(
    model,
    entry: tp.Union[str, tp.List[tp.Tuple[str, float]]],
    batch_size: int,
    device: tp.Union[torch.device, str],
    prompt_cond_key: str = "prompt",
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Encode a single chunk's prompt entry (str or weighted list) into (emb, mask).

    Routes everything through `mix_t5_prompts`, treating a plain string as a
    single (prompt, 1.0) pair. Broadcasts the entry across the batch.
    """
    if isinstance(entry, str):
        weighted: tp.List[tp.Tuple[str, float]] = [(entry, 1.0)]
    else:
        weighted = [(str(p), float(w)) for p, w in entry]
    per_batch = [weighted for _ in range(batch_size)]
    t5 = model.conditioner.conditioners[prompt_cond_key]
    assert isinstance(t5, T5Conditioner), (
        f"time_varying_prompts requires conditioner '{prompt_cond_key}' to be a "
        f"T5Conditioner (or subclass), got {type(t5).__name__}"
    )
    return mix_t5_prompts(t5, per_batch, device)


def _dominant_prompt(entry: tp.Union[str, tp.List[tp.Tuple[str, float]]]) -> str:
    """Return the dominant prompt string for a time-varying entry.

    For a plain string, returns it. For a weighted list, returns the prompt with
    the largest weight (first-wins on ties). Used to detect when a cache-refresh
    re-bootstrap is needed across chunks.
    """
    if isinstance(entry, str):
        return entry
    best_p, best_w = None, float("-inf")
    for p, w in entry:
        if w > best_w:
            best_p, best_w = str(p), float(w)
    return best_p


def _validate_time_varying_prompts(
    time_varying_prompts: tp.Optional[tp.Dict[int, tp.Any]],
    conditioning: tp.Optional[dict],
    prompt_cond_key: str,
) -> None:
    """Enforce mutual exclusion with a static prompt in `conditioning`."""
    if time_varying_prompts is None:
        return
    if conditioning is not None and prompt_cond_key in conditioning:
        raise ValueError(
            f"time_varying_prompts is mutually exclusive with conditioning['{prompt_cond_key}']; "
            "provide exactly one."
        )

def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    # Now the generative AI part:

    diff_objective = model.diffusion_objective

    if diff_objective in ["v", "v_denoiser"]:    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, device=device)

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        use_kv_cache: bool = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    # Initialize KV cache if enabled
    kv_cache = None
    if use_kv_cache:
        kv_cache = {
            'initialized': False,
        }
        conditioning_inputs['use_kv_cache'] = True
        conditioning_inputs['kv_cache'] = kv_cache

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective in ["v", "v_denoiser"]:    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_cond_inpaint(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        return_latents = False,
        use_kv_cache: bool = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion inpainting model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A mask to use for inpainting. Shape should be [batch_size, sample_size]
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    # Initialize KV cache if enabled
    kv_cache = None
    if use_kv_cache:
        kv_cache = {
            'initialized': False,
        }
        conditioning_inputs['use_kv_cache'] = True
        conditioning_inputs['kv_cache'] = kv_cache

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective in ["v", "v_denoiser"]:
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_cond_blockar(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        return_latents = False,
        block_size: int = 98304,
        generation_length: int = 2097152,
        use_kv_cache: bool = False,
        context_router: bool = False,
        context_router_attention_pattern: tp.Optional[str] = None,
        speedtest: bool = False,
        use_rope_shift: bool = False,
        rope_mode: tp.Optional[str] = None,
        plus_plus: bool = False,
        time_varying_prompts: tp.Optional[tp.Dict[int, tp.Union[str, tp.List[tp.Tuple[str, float]]]]] = None,
        prompt_cond_key: str = "prompt",
        eject_buffer_n: int = 0,
        accomp_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        accomp_cutoff: tp.Optional[int] = None,
        accomp_lead: bool = False,
        **sampler_kwargs
    ) -> torch.Tensor:
    '''
    Block-wise autoregressive ("block-AR") generation for Live Music Diffusion models.

    Audio is produced one block at a time using a sliding context window: the model is
    conditioned on the previous `sample_size - block_size` latent frames and denoises the
    next `block_size` frames. The window then slides right by `block_size` and the process
    repeats until `generation_length` samples have been produced. This is the inference-time
    counterpart to the streaming context-dropout used during training.

    Generation loop (outpaint style):
      1) Initialize the context with `init_audio` (its last `sample_size - block_size` frames)
         or zeros, and mask so only the rightmost `block_size` frames are generated.
      2) Denoise the rightmost block.
      3) Slide the context left by `block_size`, appending the freshly generated block.
      4) Repeat 2-3 until `generation_length` is reached.

    Key arguments:
      block_size:                Audio samples generated per step (converted to latent frames internally).
      generation_length:         Total audio samples to generate (>= sample_size).
      use_kv_cache:              Reuse cached context K/V across blocks for fast streaming inference.
      context_router:                   Enable context-router self-attention masking over the context window.
      context_router_attention_pattern: "enc-dec" (bidirectional context) or "block-causal" (sliding-window
                                 causal). Must match how the model was trained.
      rope_mode:                 RoPE handling for the sliding window ("fixed" / "offset" / None).
      time_varying_prompts:      Optional {block_idx: prompt | [(prompt, weight), ...]} to change the
                                 text conditioning mid-stream.
      accomp_audio / accomp_cutoff / accomp_lead:
                                 Accompaniment conditioning for live-jamming: condition generation on a
                                 provided accompaniment track, optionally with a latency cutoff and lead offset.
      init_audio:                Optional (sample_rate, tensor) to seed the initial context.
      return_latents:            Return latent frames instead of decoded audio.

    The interior per-block sampling loop mirrors generate_diffusion_cond.
    '''

    generated_audio = []
    total_generated = 0

    audio_sample_size = sample_size

    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        block_size = block_size // model.pretransform.downsampling_ratio
        generation_length = generation_length // model.pretransform.downsampling_ratio
    
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    _validate_time_varying_prompts(time_varying_prompts, conditioning, prompt_cond_key)
    if conditioning_tensors is None:
        if time_varying_prompts is not None:
            cond_for_encoding = [{k: v for k, v in conditioning[i].items() if k != prompt_cond_key} for i in range(len(conditioning))]
            # input dummy prompt so this next call doesn't throw an error
            for cond in cond_for_encoding:
                cond[prompt_cond_key] = "" 
            conditioning_tensors = model.conditioner(cond_for_encoding, device)
            # Seed prompt slot with chunk 0's entry so downstream inpaint setup has a valid tensor
            if 0 not in time_varying_prompts:
                raise ValueError("time_varying_prompts must contain an entry for chunk 0")
            emb, msk = _encode_chunk_prompt(model, time_varying_prompts[0], batch_size, device, prompt_cond_key)
            conditioning_tensors[prompt_cond_key] = [emb, msk]
        else:
            conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}


    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio.to(next(model.pretransform.parameters()).dtype))
            
        init_audio = init_audio.repeat(batch_size, 1, 1)

    # Accompaniment latents (live-jamming conditioning). When provided, the full accompaniment
    # is encoded once and sliced per chunk into the model's accomp_input/accomp_mask conditioning.
    full_accomp = None
    if accomp_audio is not None:
        accomp_sr, accomp_tensor = accomp_audio
        accomp_io_channels = model.io_channels
        if model.pretransform is not None:
            accomp_io_channels = model.pretransform.io_channels
        accomp_target_audio_len = generation_length * (model.pretransform.downsampling_ratio if model.pretransform is not None else 1)
        accomp_tensor = prepare_audio(
            accomp_tensor, in_sr=accomp_sr, target_sr=model.sample_rate,
            target_length=accomp_target_audio_len, target_channels=accomp_io_channels, device=device,
        )
        if model.pretransform is not None:
            accomp_tensor = model.pretransform.encode(accomp_tensor.to(next(model.pretransform.parameters()).dtype))
        full_accomp = accomp_tensor.repeat(batch_size, 1, 1)
        # Pad to at least generation_length latents (so per-chunk slicing always has data).
        if full_accomp.shape[-1] < generation_length:
            pad = torch.zeros(batch_size, full_accomp.shape[1], generation_length - full_accomp.shape[-1], device=device, dtype=full_accomp.dtype)
            full_accomp = torch.cat([full_accomp, pad], dim=-1)
        if accomp_cutoff is None:
            accomp_cutoff = sample_size - block_size  # default: full context window
        assert accomp_cutoff <= sample_size, f"accomp_cutoff ({accomp_cutoff}) must be <= sample_size ({sample_size})"
        print(f"Accompaniment loaded: full_accomp shape {tuple(full_accomp.shape)}, cutoff={accomp_cutoff}")

    context_latents = sample_size - block_size

    def _set_accomp(chunk_idx_local):
        """Slice the accomp window for the given chunk_idx and set conditioning_tensors. No-op
        when full_accomp is None.

        Default (accomp_lead=False): generated audio aligns with accomp[0:]. The first context
        frames of the chunk-0 window are zero-padded (no accomp data prior to t=0). For chunk c
        the window covers accomp positions [c*bs - context, c*bs - context + sample_size).

        accomp_lead=True: generated audio aligns with accomp[context:]. The first context frames
        of accomp serve as model conditioning. Window covers [c*bs, c*bs + sample_size). This
        matches the training-time alignment (target stem is paired with accomp at the same
        absolute offsets).
        """
        if full_accomp is None:
            return
        start = chunk_idx_local * block_size if accomp_lead else (chunk_idx_local * block_size - context_latents)
        # Build window with explicit zero-padding for any portion outside [0, len(full_accomp)).
        accomp_window = torch.zeros(batch_size, full_accomp.shape[1], sample_size, device=device, dtype=full_accomp.dtype)
        src_start = max(start, 0)
        src_end = min(start + sample_size, full_accomp.shape[-1])
        if src_end > src_start:
            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start)
            accomp_window[..., dst_start:dst_end] = full_accomp[..., src_start:src_end]
        # Cutoff: zero positions [cutoff:] in the window (latency horizon).
        accomp_window[..., accomp_cutoff:] = 0
        accomp_mask_window = torch.zeros(batch_size, 1, sample_size, device=device, dtype=accomp_window.dtype)
        # accomp_mask_window[..., :accomp_cutoff] = 1.0
        # Front-pad region (when start < 0, only in non-lead mode for early chunks): no real
        # accomp data, so mask = 0 there.
        # if start < 0:
        #     front_pad = min(-start, sample_size)
        #     accomp_mask_window[..., :front_pad] = 0
        conditioning_tensors['accomp_input'] = [accomp_window]
        conditioning_tensors['accomp_mask'] = [accomp_mask_window]

    def _mix_with_accomp(generated):
        """Mix the decoded generated audio with the (decoded) full accompaniment at the correct
        offset. No-op when no accomp or returning latents.

        - accomp_lead=False: generation aligns with accomp[0:]; mix directly.
        - accomp_lead=True: generation aligns with accomp[context:]; prepend zero padding of length
          context_audio_len to the generation channel, then mix.
        - Truncates both to whichever is shorter.
        """
        if full_accomp is None or return_latents:
            return generated
        accomp_audio_dec = full_accomp
        if model.pretransform is not None:
            accomp_audio_dec = model.pretransform.decode(full_accomp.to(next(model.pretransform.parameters()).dtype))
        if accomp_lead:
            ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1
            lead_pad_len = context_latents * ratio
            lead_pad = torch.zeros(generated.shape[0], generated.shape[1], lead_pad_len, device=generated.device, dtype=generated.dtype)
            generated = torch.cat([lead_pad, generated], dim=-1)
        common_len = min(generated.shape[-1], accomp_audio_dec.shape[-1])
        return generated[..., :common_len] + accomp_audio_dec[..., :common_len].to(generated.dtype)

    mask = torch.zeros((batch_size, 1, sample_size), device=device)
    print(f"Block size: {block_size}, Sample size: {sample_size}")
    mask[:, :, :sample_size - block_size] = 1.0  # condition on the leftmost sample_size - block_size samples

    if init_audio is not None:
        # truncate init_audio to the first sample_size - block_size samples
        init_audio = init_audio[:, :, : (sample_size - block_size)]
        # add initial to generated_audio
        generated_audio.append(init_audio.detach())
        # pad init_audio to sample_size with zeros on the right
        init_audio = torch.cat([init_audio, torch.zeros((batch_size, model.io_channels, block_size), device=device)], dim=2)
        inpaint_input = init_audio
    else:
        # Zero context for the initial inpaint input.
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    _set_accomp(0)
    if context_router:
        print("Using context-router attention with block size", block_size)
        sampler_kwargs["context_router_mask"] = (1 - mask)
        assert torch.all(sampler_kwargs["context_router_mask"][..., :sample_size - block_size] == 0), f"context_router_mask should be 0 for the first sample_size - block_size samples, but got {sampler_kwargs['context_router_mask'][..., :sample_size+1 - block_size]}"
        if context_router_attention_pattern is not None:
            fixed_mask_size = sample_size - block_size
            seq_len = sample_size
            assert fixed_mask_size is not None, "fixed_mask_size must be specified in inpaint_mask_kwargs when using context_router"
            print(f"Creating context-router self attention block mask with pattern {context_router_attention_pattern} and fixed mask size {fixed_mask_size} for sequence length {seq_len}")
            match context_router_attention_pattern:
                case "enc-dec":
                    # if "postpend" in sampler_kwargs and sampler_kwargs["postpend"]:
                    #     # we're moving the prepend cond to the end of the sequnce, so we need to roll the sequence by 1
                    #     # basically like a modulo operation, the last position should be like the whole prefix
                    #     def prefix_mask(b, h, q_idx, kv_idx):
                    #         return (kv_idx + 1) % (seq_len+1) < fixed_mask_size

                    #     def postfix_mask(b, h, q_idx, kv_idx):
                    #         return (q_idx + 1) % (seq_len+1) >= fixed_mask_size
                    # else:
                    def prefix_mask(b, h, q_idx, kv_idx):
                        return kv_idx < fixed_mask_size
                    
                    def postfix_mask(b, h, q_idx, kv_idx):
                        return q_idx >= fixed_mask_size
                case "block-causal":
                    bc_chunk_size = seq_len - fixed_mask_size
                    bc_n_window_chunks = fixed_mask_size // bc_chunk_size
                    bc_n_total_chunks = bc_n_window_chunks + 1
                    print(f"Block-causal context-router attention: chunk size {bc_chunk_size}, number of window chunks {bc_n_window_chunks}, total chunks {bc_n_total_chunks}")
                    def block_causal_mask(b, h, q_idx, kv_idx):
                        q_block = (q_idx // bc_chunk_size).clamp(max=bc_n_total_chunks - 1)
                        kv_block = (kv_idx // bc_chunk_size).clamp(max=bc_n_total_chunks - 1)
                        return (kv_block <= q_block) & (kv_block >= q_block - bc_n_window_chunks + 1)

            mask_mod = block_causal_mask if context_router_attention_pattern == "block-causal" else or_masks(prefix_mask, postfix_mask) 
            # create the mask now
            
            
            sampler_kwargs["self_attention_block_mask"] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len+1, KV_LEN=seq_len+1, device=noise.device, _compile=True)
            print('Created self attention block mask:')
            print(sampler_kwargs["self_attention_block_mask"].to_string())

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    model_dtype = next(model.model.parameters()).dtype

    # ── Block-causal streaming inference (KV cache persists across chunks) ──
    if context_router_attention_pattern == "block-causal" and use_kv_cache and context_router:
        chunk_size = block_size  # in latent frames
        n_window_chunks = (sample_size - block_size) // block_size
        n_total_chunks = n_window_chunks + 1
        print(f"Block-causal streaming: chunk_size={chunk_size}, n_window_chunks={n_window_chunks}, n_total_chunks={n_total_chunks}")

        sampler_type = sampler_kwargs.get("sampler_type", "euler")
        # assert sampler_type == "euler", f"Block-causal streaming only supports euler sampler, got {sampler_type}"

        # Create streaming cache and eagerly init first-step mask (avoid create_block_mask inside compiled code)
        streaming_cache = BlockCausalStreamingCache(
            model.model, chunk_size, n_window_chunks, rope_mode=rope_mode,
            accomp_cutoff=accomp_cutoff,
        )
        kv_cache = streaming_cache.init_kv_cache()
        streaming_cache.get_first_step_mask(device)

        # Bootstrap block-causal mask (same formula as training in diffusion.py)
        def block_causal_mask_fn(b, h, q_idx, kv_idx):
            q_block = (q_idx // chunk_size).clamp(max=n_total_chunks - 1)
            kv_block = (kv_idx // chunk_size).clamp(max=n_total_chunks - 1)
            return (kv_block <= q_block) & (kv_block >= q_block - n_window_chunks + 1)

        bootstrap_mask = create_block_mask(
            block_causal_mask_fn, B=None, H=None,
            Q_LEN=sample_size + 1, KV_LEN=sample_size + 1,  # +1 for postpend conditioning token
            device=device, _compile=True
        )
        print("Bootstrap block-causal mask:")
        print(bootstrap_mask.to_string())

        # context_router_mask: 0 for context, 1 for target (inverse of inpaint mask)
        context_router_mask = (1 - mask)

        # Timestep schedule (replicates sample_rf euler path)
        logsnr = torch.linspace(-6, 2, steps + 1)
        t_schedule = torch.sigmoid(-logsnr)
        t_schedule[0] = 1.0
        t_schedule[-1] = 0

        chunk_idx = 0
        prev_dominant = _dominant_prompt(time_varying_prompts[0]) if time_varying_prompts is not None else None
        while total_generated < generation_length:
            # Refresh time-varying prompt for this chunk (chunk 0 is already seeded above)
            is_rebootstrap = False
            if time_varying_prompts is not None and chunk_idx > 0:
                if chunk_idx not in time_varying_prompts:
                    raise ValueError(f"time_varying_prompts missing entry for chunk {chunk_idx}")
                emb, msk = _encode_chunk_prompt(
                    model, time_varying_prompts[chunk_idx], batch_size, device, prompt_cond_key
                )
                conditioning_tensors[prompt_cond_key] = [emb, msk]
                conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

                current_dominant = _dominant_prompt(time_varying_prompts[chunk_idx])
                if current_dominant != prev_dominant:
                    is_rebootstrap = True
                    print(f"[rebootstrap] chunk {chunk_idx}: dominant '{prev_dominant}' -> '{current_dominant}', re-priming KV cache")
                    # Wipe per-layer caches, pending chunk, and position_offset — next step 0
                    # will run as a full bootstrap (full sequence through bootstrap_mask).
                    kv_cache = streaming_cache.init_kv_cache()
                    # Optionally eject the N oldest context chunks by zeroing them in the
                    # inpaint input, lessening the pull of prior context on the new dominant.
                    if eject_buffer_n > 0:
                        n_zero = min(eject_buffer_n, n_window_chunks * block_size)
                        print(f"[rebootstrap] zeroing {n_zero} oldest context frames (eject_buffer_n={eject_buffer_n})")
                        inpaint_input = inpaint_input.clone()
                        inpaint_input[..., :n_zero] = 0
                        conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
                        conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
                prev_dominant = current_dominant

            noise = torch.randn([batch_size, model.io_channels, sample_size], device=device).type(model_dtype)
            x = noise

            conditioning_inputs_typed = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

            # if speedtest
            if speedtest:
                # do separate speedtess for chunk 0 (bootstrap), first step, and subsequent steps
                with torch.no_grad():
                    # Bootstrap step (full sequence with block-causal mask)
                    t_curr = t_schedule[0]
                    t_prev = t_schedule[1]
                    t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=device)
                    dt = t_prev - t_curr

                    # Build per-step kwargs
                    step_kwargs = dict(conditioning_inputs_typed)
                    step_kwargs.update(negative_conditioning_tensors if negative_conditioning_tensors else {})
                    step_kwargs['cfg_scale'] = cfg_scale
                    step_kwargs['batch_cfg'] = True
                    step_kwargs['rescale_cfg'] = True
                    step_kwargs['context_router_mask'] = context_router_mask
                    step_kwargs['use_kv_cache'] = True
                    step_kwargs['kv_cache'] = kv_cache

                    is_fused_first_step = False
                    rope_positions = streaming_cache.get_rope_positions(x.shape[0], device, first_step=is_fused_first_step)
                    if rope_positions is not None:
                        step_kwargs['per_item_rope_positions'] = rope_positions

                    # warmup
                    for itx in range(10):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)
                        kv_cache = streaming_cache.init_kv_cache()
                        streaming_cache.get_first_step_mask(device)
                        step_kwargs['kv_cache'] = kv_cache
                    torch.cuda.synchronize()
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for itx in range(100):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)
                        kv_cache = streaming_cache.init_kv_cache()
                        streaming_cache.get_first_step_mask(device)
                        step_kwargs['kv_cache'] = kv_cache
                    t1.record()
                    torch.cuda.synchronize()
                    print(f"Bootstrap step latency: {t0.elapsed_time(t1)/100:.2f} ms")

                    # do one more step to populate the cache for the subsequent step speedtest
                    _ = model.model(x, t_curr_tensor, **step_kwargs)
                    streaming_cache.advance(x[:, :, -block_size:])
                    # First step speed test

                    is_fused_first_step = True
                    rope_positions = streaming_cache.get_rope_positions(x.shape[0], device, first_step=is_fused_first_step)
                    if rope_positions is not None:
                        step_kwargs['per_item_rope_positions'] = rope_positions
                    # warmup
                    step_kwargs.update(streaming_cache.get_first_step_kwargs(device))
                    for itx in range(10):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)
                    torch.cuda.synchronize()
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for itx in range(100):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)
                    t1.record()
                    torch.cuda.synchronize()
                    print(f"First step latency: {t0.elapsed_time(t1)/100:.2f} ms")

                    streaming_cache.finalize_first_step()

                    # Build per-step kwargs
                    step_kwargs = dict(conditioning_inputs_typed)
                    step_kwargs.update(negative_conditioning_tensors if negative_conditioning_tensors else {})
                    step_kwargs['cfg_scale'] = cfg_scale
                    step_kwargs['batch_cfg'] = True
                    step_kwargs['rescale_cfg'] = True
                    step_kwargs['context_router_mask'] = context_router_mask
                    step_kwargs['use_kv_cache'] = True
                    step_kwargs['kv_cache'] = kv_cache

                    is_fused_first_step = False
                    rope_positions = streaming_cache.get_rope_positions(x.shape[0], device, first_step=is_fused_first_step)
                    if rope_positions is not None:
                        step_kwargs['per_item_rope_positions'] = rope_positions
                    # warmup
                    for itx in range(10):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)

                    torch.cuda.synchronize()
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for itx in range(100):
                        _ = model.model(x, t_curr_tensor, **step_kwargs)
                    t1.record()
                    torch.cuda.synchronize()
                    print(f"Subsequent step latency: {t0.elapsed_time(t1)/100:.2f} ms")
                    return
            



            for i, (t_curr, t_prev) in enumerate(zip(t_schedule[:-1], t_schedule[1:])):
                # Inpainting: replace context portion of x with noised clean audio
                noised_masked_input = inpaint_input * (1 - t_curr) + torch.randn_like(x) * t_curr
                x = x * (1 - mask) + noised_masked_input * mask

                t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=device)
                dt = t_prev - t_curr

                # Build per-step kwargs
                step_kwargs = dict(conditioning_inputs_typed)
                step_kwargs.update(negative_conditioning_tensors if negative_conditioning_tensors else {})
                step_kwargs['cfg_scale'] = cfg_scale
                step_kwargs['batch_cfg'] = True
                step_kwargs['rescale_cfg'] = True
                step_kwargs['context_router_mask'] = context_router_mask
                step_kwargs['use_kv_cache'] = True
                step_kwargs['kv_cache'] = kv_cache
                step_kwargs['plus_plus'] = plus_plus

                # With time-varying prompts, cross-attn K,V change per chunk. Disable
                # their per-module cache so they're recomputed from `context` each forward.
                if time_varying_prompts is not None:
                    step_kwargs['cross_attn_no_cache'] = True

                # RoPE positions for the current window — fused first step spans an extra chunk.
                # A re-bootstrap chunk behaves like chunk 0 (no fused step).
                use_bootstrap = (chunk_idx == 0) or is_rebootstrap
                is_fused_first_step = (i == 0 and not use_bootstrap)
                rope_positions = streaming_cache.get_rope_positions(x.shape[0], device, first_step=is_fused_first_step)
                if rope_positions is not None:
                    step_kwargs['per_item_rope_positions'] = rope_positions

                if i == 0:
                    if use_bootstrap:
                        # Bootstrap: full sequence with block-causal mask (chunk 0 or re-bootstrap)
                        step_kwargs['self_attention_block_mask'] = bootstrap_mask
                    else:
                        # Fused first step: [pending | target] with streaming mask
                        step_kwargs.update(streaming_cache.get_first_step_kwargs(device))

                v = model.model(x, t_curr_tensor, **step_kwargs)
                if plus_plus and cfg_scale != 1.0:
                    cond_v, uncond_v = v
                    cfg_v = uncond_v + cfg_scale * (cond_v - uncond_v)
                    if sampler_type == "euler":
                        denoised = x - t_curr * cfg_v
                        x = denoised + t_prev * uncond_v
                    elif sampler_type == "pingpong":
                        denoised = x - t_curr * cfg_v
                        denoised_uncond = x - t_curr * uncond_v
                        x = denoised + t_prev * (torch.randn_like(x) - denoised_uncond)

                else:
                    if sampler_type == "euler":
                        x = x + dt * v
                    elif sampler_type == "pingpong":
                        denoised = x - t_curr * v
                        x = (1- t_prev) * denoised + t_prev * torch.randn_like(x)

                # Finalize fused first step (expands cache to include pending) — not on bootstrap
                if i == 0 and not use_bootstrap:
                    streaming_cache.finalize_first_step()

            # Final inpainting: replace context with clean audio
            x = x * (1 - mask) + inpaint_input * mask

            # Extract generated block (last chunk_size frames)
            generated_block = x[:, :, -block_size:]
            generated_audio.append(generated_block.detach())
            total_generated += block_size
            # print(f"Chunk {chunk_idx} generated, total: {total_generated}/{generation_length}")

            if total_generated >= generation_length:
                break

            # Advance streaming cache: eject oldest, set pending
            streaming_cache.advance(generated_block)

            # Slide window: place generated block into context, shift left
            inpaint_input = inpaint_input.detach()
            inpaint_input[..., -block_size:] = generated_block
            inpaint_input = torch.cat([
                inpaint_input[:, :, block_size:],
                torch.zeros_like(inpaint_input[:, :, :block_size])
            ], dim=2)

            # Update conditioning with new inpaint_input and slide the accomp window.
            conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
            chunk_idx += 1
            _set_accomp(chunk_idx)
            conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
            if negative_conditioning_tensors:
                negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
                negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

        # Concatenate all generated blocks and decode
        generated_audio = torch.cat(generated_audio, dim=2)
        if model.pretransform is not None and not return_latents:
            generated_audio = generated_audio.to(next(model.pretransform.parameters()).dtype)
            generated_audio = model.pretransform.decode(generated_audio)
        return _mix_with_accomp(generated_audio)

    # ── Existing non-streaming path below ──

    # Rope shift tracking: context fills up over the first context_size // block_size blocks
    context_size = sample_size - block_size
    # If init_audio was provided, context is already full; otherwise start from 0
    rope_blocks_completed = (context_size // block_size) if (init_audio is not None) else 0

    # Time-varying prompt state for the enc-dec path. Each chunk resets its KV cache from
    # scratch (no persistent streaming cache), so there's no "rebootstrap" trigger — the
    # new prompt naturally takes effect via the per-chunk cache rebuild. The only reason
    # to track the dominant is to decide when to invoke eject_buffer_n: zero the oldest N
    # context chunks so the newly-dominant prompt isn't pulled toward stale audio.
    encdec_chunk_idx = 0
    encdec_prev_dominant = _dominant_prompt(time_varying_prompts[0]) if time_varying_prompts is not None else None
    n_window_chunks_encdec = context_size // block_size

    while total_generated < generation_length:
        if time_varying_prompts is not None and encdec_chunk_idx > 0:
            if encdec_chunk_idx not in time_varying_prompts:
                raise ValueError(f"time_varying_prompts missing entry for chunk {encdec_chunk_idx}")
            emb, msk = _encode_chunk_prompt(
                model, time_varying_prompts[encdec_chunk_idx], batch_size, device, prompt_cond_key
            )
            conditioning_tensors[prompt_cond_key] = [emb, msk]

            current_dominant = _dominant_prompt(time_varying_prompts[encdec_chunk_idx])
            if current_dominant != encdec_prev_dominant and eject_buffer_n > 0:
                n_zero = eject_buffer_n
                print(f"[enc-dec] chunk {encdec_chunk_idx}: dominant '{encdec_prev_dominant}' -> '{current_dominant}', zeroing {n_zero} oldest context frames")
                inpaint_input = inpaint_input.clone()
                inpaint_input[..., :n_zero] = 0
                conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
            encdec_prev_dominant = current_dominant

            conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
            # Negative conditioning is static for time-varying prompts (we don't vary
            # the negative prompt), so no refresh needed here.

        noise = noise.type(model_dtype)
        conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

        # Initialize KV cache if enabled (reset for each block)
        kv_cache = None
        if use_kv_cache:
            # Clear per-module caches from previous block
            model.model.model.clear_kv_cache()
            kv_cache = {
                'initialized': False,
            }
            conditioning_inputs['use_kv_cache'] = True
            conditioning_inputs['kv_cache'] = kv_cache
            # model.model.model.init_kv_cache(batch_size=batch_size, encoder_seq_len=sample_size - block_size, device=device, dtype=model_dtype, context_seq_len=65)

        # Per-item rope shift: for the first context_size // block_size blocks the context
        # is not yet fully filled, so we modulate rope positions to match training.
        if use_rope_shift:
            filled_context = min(rope_blocks_completed * block_size, context_size)
            if filled_context < context_size:
                target_start = filled_context
                ctx_pos = torch.arange(context_size, device=device).unsqueeze(0).expand(batch_size, -1)
                tgt_pos = (target_start + torch.arange(block_size, device=device)).unsqueeze(0).expand(batch_size, -1)
                sampler_kwargs['per_item_rope_positions'] = torch.cat([ctx_pos, tgt_pos], dim=1).long()
            else:
                sampler_kwargs.pop('per_item_rope_positions', None)

        # k-diffusion denoising process go!
        diff_objective = model.diffusion_objective

        if diff_objective in ["v", "v_denoiser"]:    
            # k-diffusion denoising process go!
            sampled = sample_k(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
        elif diff_objective in ["rectified_flow", "rf_denoiser"]:

            if "sigma_min" in sampler_kwargs:
                del sampler_kwargs["sigma_min"]

            if "rho" in sampler_kwargs:
                del sampler_kwargs["rho"]

            if speedtest:
                n_warmup = 10
                n_iters = 100
                # warmup
                for _ in range(n_warmup):
                    _ = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, plus_plus=plus_plus, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for _ in range(n_iters):
                    _ = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, plus_plus=plus_plus, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
                t1.record()
                torch.cuda.synchronize()
                print(f"Average inference time per block: {t0.elapsed_time(t1) / n_iters} ms")
                return

            else:
                sampled = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, plus_plus=plus_plus, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
        
       # Get the last block_size samples from sampled
        generated_block = sampled[:, :, -block_size:]
        generated_audio.append(generated_block.detach())
        total_generated += block_size
        if total_generated >= generation_length:
            del sampled
            del conditioning_tensors
            del conditioning_inputs
            torch.cuda.empty_cache()
            break
        # update inpaint_input
        inpaint_input = inpaint_input.detach()
        inpaint_input[..., -block_size:] = generated_block
        # slide inpaint_input to the left by block_size samples
        inpaint_input = torch.cat([inpaint_input[:, :, block_size:], torch.zeros_like(inpaint_input[:, :, :block_size])], dim=2)
        # mask stays the same
        conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        _set_accomp(encdec_chunk_idx + 1)
        conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
        if negative_conditioning_tensors:
            negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
            negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
        rope_blocks_completed += 1
        encdec_chunk_idx += 1
        del sampled

    generated_audio = torch.cat(generated_audio, dim=2)
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        generated_audio = generated_audio.to(next(model.pretransform.parameters()).dtype)
        generated_audio = model.pretransform.decode(generated_audio)

    # Return audio
    return _mix_with_accomp(generated_audio)
        


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask