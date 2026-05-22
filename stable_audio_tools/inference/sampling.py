import torch
import math
from tqdm import trange, tqdm
import torch.distributions as dist

import k_diffusion as K

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def time_shift(self, t: torch.Tensor, seq_len: int):
        seq_len = min(max(seq_len,self.min_length), self.max_length)
        sigma = 1.0
        mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
        t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

        if self.use_sine:
            t_out = torch.sin(t_out * math.pi / 2)

        return t_out

def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        mean_logsnr (float): Mean of the logSNR Gaussian distribution
        std_logsnr (float): Standard deviation of the logSNR Gaussian distribution

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from Gaussian distribution
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t
def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):
    """

    shape: shape of the output tensor
    left_trunc: left truncation point, fraction of probability to be discarded
    right_trunc: right truncation boundary, should be 1 (never seen at test time)
    """

    # Step 1: Sample from the logistic normal distribution (sigmoid of normal)
    logits = torch.randn(shape)

    # Step 2: Apply the CDF transformation of the normal distribution
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)

    # Step 3: Define the truncation bounds on the CDF
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    # Step 4: Rescale linear CDF values into the truncated region (between lower_bound and upper_bound)
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values

    # Step 5: Map back to logistic-normal space using inverse CDF
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))

    # Step 6: Rescale values so that min is 0 and max is just below 1
    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

@torch.no_grad()
def sample_discrete_euler(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, disable_tqdm=True, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. Euler method"""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    #alphas, sigmas = 1-t, t

    plus_plus = extra_args.get("plus_plus", False)
    cfg_scale = extra_args.get("cfg_scale", 1.0)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        # Broadcast the current timestep to the correct shape
        if inpaint_mask is not None and inpaint_masked_input is not None:
            # add noise to the masked input according to the current sigma
            noised_masked_input = inpaint_masked_input * (1 - t_curr) + torch.randn_like(x) * t_curr
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )

        dt = t_prev - t_curr  # we solve backwards in our formulation

        v = model(x, t_curr_tensor, **extra_args)
        if plus_plus and cfg_scale != 1.0:
            cond_v, uncond_v = v
            cfg_v = uncond_v + cfg_scale * (cond_v - uncond_v)
            denoised = x - t_curr * cfg_v
            x = denoised + t_prev * uncond_v
        else:
            x = x + dt * v

        if callback is not None:
            v_for_cb = (cond_v if plus_plus and cfg_scale != 1.0 else v)
            denoised = x - t_prev * v_for_cb
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    if inpaint_mask is not None and inpaint_masked_input is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask

    return x

@torch.no_grad()
def sample_rk4(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta"""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:
        
        # Create the noise schedule
        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    #alphas, sigmas = 1-t, t

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]))):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # we solve backwards in our formulation

        k1 = model(x, t_curr_tensor, **extra_args)
        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)
        k4 = model(x + dt * k3, t_prev * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if callback is not None:
            denoised = x - t_prev * k4
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    return x

@torch.no_grad()
def sample_flow_dpmpp(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, disable_tqdm=True,  inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models"""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    
    else:
        t = sigmas

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()

    for i in trange(len(t) - 1, disable=disable_tqdm):

        if inpaint_mask is not None and inpaint_masked_input is not None:
            # add noise to the masked input according to the current sigma
            noised_masked_input = inpaint_masked_input * (1 - t[i]) + torch.randn_like(x) * t[i]
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask

        t_curr, t_next = t[i], t[i + 1]

        denoised = x - t_curr * model(x, t_curr * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'denoised': denoised})
        alpha_t = 1-t_next
        h = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or t_next == 0:
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised_d
        old_denoised = denoised

    if inpaint_mask is not None and inpaint_masked_input is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask
    return x

@torch.no_grad()
def sample_flow_pingpong(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for distilled models"""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    
    else:
        t = sigmas

    plus_plus = extra_args.get("plus_plus", False)
    cfg_scale = extra_args.get("cfg_scale", 1.0)

    for i in trange(len(t) - 1, disable=True):

        if inpaint_mask is not None and inpaint_masked_input is not None:
            # add noise to the masked input according to the current sigma
            noised_masked_input = inpaint_masked_input * (1 - t[i]) + torch.randn_like(x) * t[i]
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask

        v = model(x, t[i] * ts, **extra_args)
        t_next = t[i + 1]
        if plus_plus and cfg_scale != 1.0:
            cond_v, uncond_v = v
            cfg_v = uncond_v + cfg_scale * (cond_v - uncond_v)
            denoised = x - t[i] * cfg_v
            denoised_uncond = x - t[i] * uncond_v
            if callback is not None:
                callback({'x': x, 'i': i, 't': t[i], 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
            x = denoised + t_next * (torch.randn_like(x) - denoised_uncond)
        else:
            denoised = x - t[i] * v
            if callback is not None:
                callback({'x': x, 'i': i, 't': t[i], 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
            x = (1-t_next) * denoised + t_next * torch.randn_like(x)

    if inpaint_mask is not None and inpaint_masked_input is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask

    return x


def v_alpha_for_sigma(sigma):
    return (1 - sigma**2).sqrt()

def v_model_t_for_sigma(sigma):
    return  (2 / math.pi) * torch.arcsin(sigma.clamp(0, 1))

@torch.no_grad()
def sample_v_pingpong(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for v-diffusion"""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    logsnr = torch.linspace(-6, 2, steps + 1)
    sigmas = torch.sigmoid(-logsnr)
    sigmas[0] = 1.0
    sigmas[-1] = 0.0
    # t = (2 / math.pi) * torch.arcsin(sigmas.clamp(0, 1))

    # if dist_shift is not None:
    #     t = dist_shift.time_shift(t, x.shape[-1])
    

    # alphas, sigmas = get_alphas_sigmas(t)

    for i in trange(steps, disable=True):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1] 
        alpha_curr = v_alpha_for_sigma(sigma_curr)
        t_model = v_model_t_for_sigma(sigma_curr)

        if inpaint_mask is not None and inpaint_masked_input is not None:
            # add noise to the masked input according to the current sigma
            noised_masked_input = inpaint_masked_input * alpha_curr + torch.randn_like(x) * sigma_curr
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask

        v = model(x, t_model * ts, **extra_args)
        # denoised = (1-t[i]) * x - t[i] * v
        denoised = alpha_curr * x - sigma_curr * v
        if callback is not None:
            callback({'x': x, 'i': i, 't': t_model, 'sigma': sigma_curr, 'sigma_hat': sigma_curr, 'denoised': denoised})

        # t_next = t[i + 1]
        # x = (1-t_next) * denoised + t_next * torch.randn_like(x)
        if i < steps - 1:
            alpha_next = v_alpha_for_sigma(sigma_next)
            x = alpha_next * denoised + sigma_next * torch.randn_like(x)
        else:
            x = denoised

    if inpaint_mask is not None and inpaint_masked_input is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask

    return x

@torch.no_grad()
def sample(model, x, steps, eta, callback=None, sigma_max=1.0, dist_shift=None, cfg_pp=False, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)[:-1]

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):
        if inpaint_mask is not None and inpaint_masked_input is not None:
            # add noise to the masked input according to the current sigma
            noised_masked_input = inpaint_masked_input * alphas[i] + torch.randn_like(x) * sigmas[i]
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask

        if cfg_pp:
            # Get the model output (v, the predicted velocity)
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        # Predict the noise and the denoised data
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        # If we are not on the last timestep, compute the noisy data for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised data in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

        if callback is not None:
            denoised = pred
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    if inpaint_mask is not None and inpaint_masked_input is not None:
        pred = pred * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask
    return pred

def sample_v_dpmpp(model, x, sigmas=None, steps=None, callback=None, dist_shift=None, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """DPM-Solver++ for v-diffusion models. Returns output at each step.

    Args:
        sigmas: v-diffusion timestep schedule, from 1 (noise) to 0 (clean).
                In v-diffusion: alpha(t) = cos(t*pi/2), sigma(t) = sin(t*pi/2).
    """
    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    ts = x.new_ones([x.shape[0]])

    if sigmas is None:
        t = torch.linspace(1, 0, steps + 1)
        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    else:
        t = sigmas

    old_denoised = None

    get_alpha_sigma = lambda tv: (torch.cos(tv * math.pi / 2), torch.sin(tv * math.pi / 2))
    log_snr = lambda tv: (torch.cos(tv * math.pi / 2) / torch.sin(tv * math.pi / 2)).clamp(min=1e-20).log()
    for i in range(len(t) - 1):
        alpha_i, sigma_i = get_alpha_sigma(t[i])

        if inpaint_masked_input is not None and inpaint_mask is not None:
            noised_masked_input = inpaint_masked_input * alpha_i + torch.randn_like(inpaint_masked_input) * sigma_i
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask



        # v-prediction: v = alpha * noise - sigma * x_0  =>  x_0 = alpha * x - sigma * v
        v = model(x, t[i] * ts, **extra_args)
        denoised = alpha_i * x - sigma_i * v

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_i, 'sigma_hat': sigma_i, 'denoised': denoised})

        t_curr, t_next = t[i], t[i + 1]
        alpha_next, sigma_next = get_alpha_sigma(t_next)
        h = log_snr(t_next) - log_snr(t_curr)

        if old_denoised is None or t_next == 0:
            x = (sigma_next / sigma_i) * x - alpha_next * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_next / sigma_i) * x - alpha_next * (-h).expm1() * denoised_d
        old_denoised = denoised
        # assert x is not nan or any other bad value
        assert torch.isfinite(x).all(), f"Non-finite value encountered in sample_v_dpmpp_w_intermediates at step {i}: {x}"

    if inpaint_masked_input is not None and inpaint_mask is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask
    target = x.detach()

    return target


# Soft mask inpainting is just shrinking hard (binary) mask inpainting
# Given a float-valued soft mask (values between 0 and 1), get the binary mask for this particular step
def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    # convert to binary mask
    bmask = torch.where(mask<=strength,1,0)
    return bmask

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, init_data to none
# For variations, set init_data
def sample_k(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.01,
        sigma_max=100,
        rho=1.0,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]
    is_v_diff = sampler_type in ["v-ddim", "v-ddim-cfgpp", "v-dpmpp", 'v-pingpong']

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise


        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
    elif is_v_diff:

        if sigma_max > 1: # sigma_max should be between 0 and 1
            sigma_max = 1

        if cond_fn is not None:
            model_fn = make_cond_model_fn(model_fn, cond_fn)

        alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))

        if init_data is not None:
            x = init_data * alpha + noise * sigma
        else:
            x = noise

        if sampler_type == "v-ddim" or sampler_type == "v-ddim-cfgpp":
            use_cfg_pp = sampler_type == "v-ddim-cfgpp"
            return sample(model_fn, x, steps, eta=0.0, sigma_max=sigma_max, cfg_pp=use_cfg_pp, callback=callback, **extra_args)
        elif sampler_type == "v-dpmpp":
            return sample_v_dpmpp(model_fn, x, sigmas=None, steps=steps, callback=callback, **extra_args)
        elif sampler_type == "v-pingpong":
            return sample_v_pingpong(model_fn, x, sigmas=None, steps=steps, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")

# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data
def sample_rf(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="euler",
        sigma_max=1,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    if sigma_max > 1:
        sigma_max = 1

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    if init_data is not None:

        # VARIATION
        # Interpolate the init data and the noise for init audio
        x = init_data * (1 - sigma_max) + noise * sigma_max

    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise

    logsnr_max = math.log(((1-sigma_max)/sigma_max) + 1e-6) if sigma_max < 1 else -6

    logsnr = torch.linspace(logsnr_max, 2, steps + 1)

    t = torch.sigmoid(-logsnr)

    t[0] = sigma_max
    t[-1] = 0

    if sampler_type == "euler":
        return sample_discrete_euler(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    elif sampler_type == "rk4":
        return sample_rk4(model_fn, x, steps, sigma_max, callback=callback, **extra_args)
    elif sampler_type == "dpmpp":
        return sample_flow_dpmpp(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    elif sampler_type == "pingpong":
        return sample_flow_pingpong(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
