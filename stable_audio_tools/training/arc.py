import gc
import math
import pytorch_lightning as pl
import random
import torch
import typing as tp

from ema_pytorch import EMA
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, or_masks
import torch.distributed as dist

from ..inference.sampling import truncated_logistic_normal_rescaled
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..inference.generation import _dominant_prompt
from ..models.conditioners import T5Conditioner, mix_t5_prompts
from ..models.dit import BlockCausalStreamingCache
from ..models.inpainting import random_inpaint_mask, apply_accomp_cutoff
from .utils import create_optimizer_from_config, create_scheduler_from_config

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def euler_step(x_t, v_t, t, s):
    return x_t + (s - t)[:, None, None] * v_t

@torch.no_grad()
def sample_flow_dpmpp_w_intermediates(model, x, sigmas=None, steps=None, callback=None, dist_shift=None, inpaint_masked_input=None, inpaint_mask=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models. Return output at each step."""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(1, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    
    else:
        t = sigmas

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()
    inters_x = []
    inters_t = []

    for i in range(len(t) - 1):
        if inpaint_masked_input is not None and inpaint_mask is not None:
            noised_masked_input = inpaint_masked_input * (1-t[i]) + torch.randn_like(inpaint_masked_input) * t[i]
            x = x * (1 - inpaint_mask) + noised_masked_input * inpaint_mask
        inters_x.append(x)
        inters_t.append(t[i])
        denoised = x - t[i] * model(x, t[i] * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
        t_curr, t_next = t[i], t[i + 1]
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

    if inpaint_masked_input is not None and inpaint_mask is not None:
        x = x * (1 - inpaint_mask) + inpaint_masked_input * inpaint_mask
    target = x.detach()
    inters_x = torch.stack(inters_x).detach() # steps x B x C x T
    inters_t = torch.stack(inters_t).unsqueeze(-1).detach() # steps x 1

    return {'target': target, 'x': inters_x, 't': inters_t}

class ARCTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for ARC post-training on a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            discriminator: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            teacher_model: ConditionedDiffusionModelWrapper = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None,
            accomp_kwargs = None,
    ):
        super().__init__()

        self.automatic_optimization = False

        self.diffusion = model
        self.accomp_kwargs = accomp_kwargs
        if self.accomp_kwargs is not None:
            print(f"ARC accompaniment kwargs: {self.accomp_kwargs}")

        self.teacher_model = teacher_model

        if self.teacher_model is not None:
            self.teacher_model.eval().requires_grad_(False)

        self.discriminator = discriminator

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.99,
                power=1,
                update_every=1,
                update_after_step=0,
                include_online_model=False,
                min_value=0.99,
                inv_gamma=1.0,
            )
        else:
            self.diffusion_ema = None

        self.ode_warmup_config = arc_config.get('ode_warmup', None)

        if self.ode_warmup_config is not None:

            self.ode_warmup_steps = self.ode_warmup_config.get('warmup_steps', 0)
            self.ode_refresh_rate = self.ode_warmup_config.get('refresh_rate', 1)
            self.ode_n_sampling_steps = self.ode_warmup_config.get('sampling_steps', 20)
            self.ode_warmup_cfg = self.ode_warmup_config.get('cfg', 4.0)
        else:
            self.ode_warmup_steps = 0

        # RF warmup: simple standard RF/diffusion loss on real data. Overridden by ode_warmup if both provided.
        self.rf_warmup_config = arc_config.get('rf_warmup', None)
        if self.rf_warmup_config is not None and self.ode_warmup_config is None:
            self.rf_warmup_steps = self.rf_warmup_config.get('warmup_steps', 0)
        else:
            self.rf_warmup_steps = 0

        sampling_config = arc_config.get('sampling', None)

        self.diff_states = []

        self.noise_dist_config = arc_config.get('noise_dist', {})
        self.gen_noise_dist = self.build_noise_dist('generator')
        self.dis_noise_dist = self.build_noise_dist('discriminator')

        self.discriminator_config = arc_config.get('discriminator', {})
        self.grad_penalty_weight = self.discriminator_config.get('grad_penalty_weight', 1.0)

        self.discriminator_dit_layer = self.discriminator_config.get('dit_hidden_layer', None)

        self.do_contrastive_disc = self.discriminator_config.get('contrastive', False)
        self.roll_keys = self.discriminator_config.get('roll_keys', ["prompt"])
        
        self.include_grad_penalties = self.discriminator_config.get('include_grad_penalties', False)

        assert self.discriminator_dit_layer is not None, "Must specify discriminator dit_hidden_layer in ARC config"

        discriminator_type = self.discriminator_config.get('type', 'convnext')
        discriminator_model_config = self.discriminator_config.get('config', {})


        self.discriminator_loss_type = self.discriminator_config.get('loss_type', 'relativistic')

        self.gen_gan_weight = self.discriminator_config.get('weights', {}).get('generator', 1.0)
        self.dis_gan_weight = self.discriminator_config.get('weights', {}).get('discriminator', 1.0)
        if self.gen_gan_weight > 0.0 or self.dis_gan_weight > 0.0:
            if discriminator_type == 'convnext':
                from ..models.arc import ConvNeXtDiscriminator
                self.discriminator_head = ConvNeXtDiscriminator(in_channels = self.discriminator.model.model.transformer.dim, latent_dim=1, **discriminator_model_config)
            elif discriminator_type == 'conv':
                from ..models.arc import ConvDiscriminator
                self.discriminator_head = ConvDiscriminator(channels = self.discriminator.model.model.transformer.dim, **discriminator_model_config)
        else:
            self.discriminator_head = None

        if self.do_contrastive_disc:
            self.contrastive_loss_weight = self.discriminator_config.get('weights', {}).get('contrastive', 1.0)

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.diffusion_objective = model.diffusion_objective

        assert optimizer_configs is not None, "Must specify optimizer_configs in training config"

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        # self.model_last_layer = self.diffusion.model.model.transformer.project_out.weight

        self.clip_grad_norm = clip_grad_norm

        self.trim_config = trim_config

        if self.trim_config is not None:
            self.trim_prob = self.trim_config.get("trim_prob", 0.0)
            self.trim_type = self.trim_config.get("type", "random_item")

        self.inpainting_config = inpainting_config

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            self.ode_inpaint_mask = None
            self.ode_inpaint_masked_input = None
            print(f"Inpainting mask kwargs: {self.inpaint_mask_kwargs}")

        # Validation

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def _get_gen_params(self):
        """Get generator parameters."""
        return list(self.diffusion.parameters())

    def _get_disc_params(self):
        """Get discriminator parameters (full model + head)."""
        disc_head_params = list(self.discriminator_head.parameters()) if self.discriminator_head is not None else []
        return disc_head_params + list(self.discriminator.parameters())

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        disc_opt_config = self.optimizer_configs['discriminator']
        if getattr(self, "use_kv_cache", False):
            # need to run the split qkv matrices before instantiating the optimizer to avoid compilation errors
            # loop through all submodule of model.model.model.transformer, and for the ones that are Attention, run _split_qkv_projections_for_cache()
            print("Splitting qkv projections for KV cache...")
            for submodule in self.diffusion.model.model.transformer.modules():
                if hasattr(submodule, "_split_qkv_projections_for_cache"):
                    submodule._split_qkv_projections_for_cache()

        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self._get_gen_params())
        opt_disc = create_optimizer_from_config(disc_opt_config['optimizer'], self._get_disc_params())
        if "scheduler" in diffusion_opt_config and "scheduler" in disc_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            sched_disc = create_scheduler_from_config(disc_opt_config['scheduler'], opt_disc)
            sched_disc_config = {
                "scheduler": sched_disc,
                "interval": "step"
            }
            return [opt_diff, opt_disc], [sched_diff_config, sched_disc_config]

        return [opt_diff, opt_disc]

    def rf_warmup_step(self, diffusion_input, metadata, padding_masks):
        """Standard rectified-flow loss on real data — simple supervised warmup."""
        conditioning = self.diffusion.conditioner(metadata, self.device)

        inpaint_mask = None
        inpaint_masked_input = None
        context_router = None
        block_mask = None

        if self.inpainting_config is not None:
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(
                diffusion_input, padding_masks=padding_masks,
                **self.inpaint_mask_kwargs,
            )
            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]
            if hasattr(self, "_get_block_mask"):
                block_mask = self._get_block_mask(diffusion_input.device)
                context_router = 1 - inpaint_mask

        t = self.gen_noise_dist(diffusion_input.shape[0])
        noise = torch.randn_like(diffusion_input)
        x_t = diffusion_input * (1 - t)[:, None, None] + noise * t[:, None, None]
        target_v = noise - diffusion_input

        v_pred = self.diffusion(
            x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob,
            context_router_mask=context_router, self_attention_block_mask=block_mask,
        )

        rf_loss = F.mse_loss(v_pred, target_v, reduction='none')
        if self.inpainting_config is not None:
            loss_mask = (1 - inpaint_mask).bool()
            if loss_mask.ndim == 2 and rf_loss.ndim == 3:
                loss_mask = loss_mask.unsqueeze(1)
            if loss_mask.shape[1] != rf_loss.shape[1]:
                loss_mask = loss_mask.repeat(1, rf_loss.shape[1], 1)
            rf_loss = rf_loss[loss_mask].mean()
        else:
            rf_loss = rf_loss.mean()
        return rf_loss

    def ode_warmup_step(self, diffusion_input, metadata, padding_masks):
        refresh_diff_states = self.global_step % self.ode_refresh_rate == 0

        if refresh_diff_states:
            start_noise = torch.randn_like(diffusion_input)
            teacher_conditioning = self.teacher_model.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                # Create a mask of random length for a random slice of the input
                inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

                teacher_conditioning['inpaint_mask'] = [inpaint_mask]
                teacher_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                self.ode_inpaint_mask = inpaint_mask
                self.ode_inpaint_masked_input = inpaint_masked_input

                if hasattr(self, "_get_block_mask"):
                    self.ode_block_mask = self._get_block_mask(diffusion_input.device)
                    self.ode_context_router = 1 - inpaint_mask

            logsnr = torch.linspace(-6, 2, self.ode_n_sampling_steps + 1)
            t = torch.sigmoid(-logsnr)
            t[0] = 1
            t[-1] = 0

            self.ode_metadata = metadata
            self.diff_states = sample_flow_dpmpp_w_intermediates(
                self.teacher_model, start_noise, sigmas=t, cond=teacher_conditioning, 
                cfg_scale=self.ode_warmup_cfg, dist_shift=self.teacher_model.dist_shift, batch_cfg=True, 
                inpaint_masked_input=inpaint_masked_input if self.inpainting_config is not None else None,
                inpaint_mask=inpaint_mask if self.inpainting_config is not None else None,
                context_router_mask=self.ode_context_router if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None,
                self_attention_block_mask=self.ode_block_mask if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None

            )

        conditioning = self.diffusion.conditioner(self.ode_metadata, self.device)

        if self.inpainting_config is not None:
            conditioning['inpaint_mask'] = [self.ode_inpaint_mask]
            conditioning['inpaint_masked_input'] = [self.ode_inpaint_masked_input]

        ixs = torch.randint(0, self.ode_n_sampling_steps, (diffusion_input.shape[0],))
        t = self.diff_states['t'].clone().detach()[ixs].squeeze(-1)
        x_t = self.diff_states['x'].clone().detach()[ixs.flatten(), torch.arange(diffusion_input.shape[0])].squeeze(0)

        t = t.to(self.device).detach().clone().requires_grad_(False)
        x_t = x_t.to(self.device).detach().clone().requires_grad_(False)

        v_t_student = self.diffusion(x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob,
            context_router_mask=self.ode_context_router if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None,
            self_attention_block_mask=self.ode_block_mask if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None
        )
        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))
        ode_mse_loss = F.mse_loss(denoised_student, self.diff_states['target'], reduction='none')
        if self.inpainting_config is not None:
            loss_mask = (1 - self.ode_inpaint_mask).bool()
            if loss_mask.ndim == 2 and ode_mse_loss.ndim == 3:
                loss_mask = loss_mask.unsqueeze(1)

            if loss_mask.shape[1] != ode_mse_loss.shape[1]:
                loss_mask = loss_mask.repeat(1, ode_mse_loss.shape[1], 1)
            ode_mse_loss = ode_mse_loss[loss_mask].mean() 
        else:
            ode_mse_loss = ode_mse_loss.mean()


        return ode_mse_loss

    def calculate_disc_loss(self, real_scores, fake_scores):
        if self.discriminator_loss_type == 'lsgan':
            loss_dis = (F.mse_loss(real_scores, torch.ones_like(real_scores)) + F.mse_loss(fake_scores, torch.zeros_like(fake_scores))) * self.dis_gan_weight
        elif self.discriminator_loss_type == 'relativistic':
            diff = real_scores - fake_scores
            loss_dis = F.softplus(-diff).mean() * self.dis_gan_weight
        elif self.discriminator_loss_type == 'relativistic-lsgan':
            diff = real_scores - fake_scores
            loss_dis = F.mse_loss(-diff, torch.ones_like(diff), reduction="mean") * self.dis_gan_weight
        else:
            raise ValueError(f"Unknown discriminator loss type: {self.discriminator_loss_type}")
        return loss_dis

    def calculate_gen_adv_loss(self, real_scores, fake_scores):
        if self.discriminator_loss_type == 'lsgan':
            loss_adv = F.mse_loss(fake_scores, torch.ones_like(fake_scores)) * self.gen_gan_weight
        elif self.discriminator_loss_type == 'relativistic':
            diff = real_scores - fake_scores
            loss_adv = F.softplus(diff).mean() * self.gen_gan_weight
        elif self.discriminator_loss_type == 'relativistic-lsgan':
            diff = real_scores - fake_scores
            loss_adv = F.mse_loss(diff, torch.ones_like(diff), reduction="mean") * self.gen_gan_weight
        else:
            raise ValueError(f"Unknown discriminator loss type: {self.discriminator_loss_type}")
        return loss_adv

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        if self.global_step < self.ode_warmup_steps:
            ode_mse_loss = self.ode_warmup_step(diffusion_input, metadata, padding_masks)

            opt_gen.zero_grad()
            self.manual_backward(ode_mse_loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict = {'train/ode_mse_loss': ode_mse_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            return ode_mse_loss

        if self.global_step < self.rf_warmup_steps:
            rf_loss = self.rf_warmup_step(diffusion_input, metadata, padding_masks)

            opt_gen.zero_grad()
            self.manual_backward(rf_loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict = {'train/rf_warmup_loss': rf_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            return rf_loss

        # Start trimming after ODE warmup to avoid sequence length issues
        if self.trim_config is not None:
            if random.random() < self.trim_prob:
                
                data_lengths = (torch.sum(padding_masks, dim=1) - 1).tolist()
                
                if self.trim_type == "random_item":
                    trim_length = max(random.choice(data_lengths), 128)

                diffusion_input = diffusion_input[:, :, :trim_length]

        conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.inpainting_config is not None:

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        t = self.gen_noise_dist(reals.shape[0])  

        gen_noise = torch.randn_like(diffusion_input)
        x_t = diffusion_input * (1-t)[:, None, None] + gen_noise * t[:, None, None]       
      
        train_gen = (self.global_step % 2) == 0

        if train_gen or self.global_step < self.ode_warmup_steps:
            v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob)
        else:
            x_t = x_t.requires_grad_(False)
            with torch.no_grad():
                v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning).detach()

        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            # Get discriminator scores for adversarial loss
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            x_t_gan = denoised_student * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]

            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            v_t_gan_hidden_states = self.discriminator(x_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=self.discriminator_dit_layer)

            disc_scores = self.discriminator_head(v_t_gan_hidden_states.transpose(1, 2))

            log_dict['gen_disc_scores_mean'] = disc_scores.mean().detach()
            log_dict['gen_disc_scores_std'] = disc_scores.std().detach()

            # Generator adversarial loss (relativistic variants need real scores too)
            if self.discriminator_loss_type in ('relativistic', 'relativistic-lsgan'):
                x_t_gan_real = diffusion_input * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]
                v_t_gan_real_hidden_states = self.discriminator(x_t_gan_real, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=self.discriminator_dit_layer)
                disc_scores_real = self.discriminator_head(v_t_gan_real_hidden_states.transpose(1, 2))
            else:
                disc_scores_real = None

            loss_adv = self.calculate_gen_adv_loss(disc_scores_real, disc_scores)

            loss = loss_adv

            log_dict['train/adv_loss'] = loss_adv.detach()

            opt_gen.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict['train/gen_loss'] = loss.detach()
        else:
            denoised_student = denoised_student.detach().requires_grad_(True)
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            reals_t_gan = diffusion_input * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            denoised_t_gan = denoised_student * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            
            reals_t_gan = reals_t_gan.detach().requires_grad_(True)
            denoised_t_gan = denoised_t_gan.detach().requires_grad_(True)
            
            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
            denoised_gan_hidden_states = checkpoint(self.discriminator, denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)

            disc_scores_reals = checkpoint(self.discriminator_head, reals_gan_hidden_states)
            disc_scores_denoised = checkpoint(self.discriminator_head, denoised_gan_hidden_states)

            if self.include_grad_penalties:
        
                r1_approx_variance = 0.05

                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                noised_reals_gan_hidden_states = checkpoint(self.discriminator, noised_reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
                noised_denoised_gan_hidden_states = checkpoint(self.discriminator, noised_denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
                disc_scores_noised_reals = checkpoint(self.discriminator_head, noised_reals_gan_hidden_states)
                disc_scores_noised_denoised = checkpoint(self.discriminator_head, noised_denoised_gan_hidden_states)
                r1_diff = disc_scores_noised_reals - disc_scores_reals
                r2_diff = disc_scores_noised_denoised - disc_scores_denoised

                r1_penalty = torch.sum(r1_diff ** 2, dim=[1, 2])
                r2_penalty = torch.sum(r2_diff ** 2, dim=[1, 2])

                log_dict['r1_penalty'] = r1_penalty.mean().detach()
                log_dict['r2_penalty'] = r2_penalty.mean().detach()

                grad_penalty_loss = ((r1_penalty.mean() + r2_penalty.mean())/2)

                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

           

            log_dict['disc_real_scores_mean'] = disc_scores_reals.mean().detach()
            log_dict['disc_fake_scores_mean'] = disc_scores_denoised.mean().detach()
            log_dict['disc_real_scores_std'] = disc_scores_reals.std().detach()
            log_dict['disc_fake_scores_std'] = disc_scores_denoised.std().detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised)

            if self.do_contrastive_disc:

                rolled_metadata = []

                for i in range(reals.shape[0]):
                    rolled_keys = ["prompt"]
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)

                if self.inpainting_config is not None:
                    # Hold inpainting conditioning constant during contrastive conditioning
                    rolled_conditioning['inpaint_mask'] = [inpaint_mask]
                    rolled_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                rolled_reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=rolled_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)

                disc_scores_rolled_reals = checkpoint(self.discriminator_head, rolled_reals_gan_hidden_states)

                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals) * self.contrastive_loss_weight

                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss

            log_dict['train/dis_loss'] = loss_dis.detach()

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()
            opt_disc.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()), self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                sched_disc.step()

        log_dict['train/std_data']: diffusion_input.std()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        
        return loss

    def build_noise_dist(self, key):
        dist = self.noise_dist_config.get(key, 'uniform')
        match dist:
            case 'uniform':
                return lambda b: self.rng.draw(b)[:, 0].to(self.device)
            case 'logit_normal':
                return lambda b: torch.sigmoid(torch.randn(b, device=self.device))
            case 'trunc_logit_normal':
                return lambda b: 1 - truncated_logistic_normal_rescaled(b).to(self.device)
            case 'one_shot':
                return lambda b: torch.ones(b, device=self.device)
            case 'denoised':
                return lambda b: torch.zeros(b, device=self.device)
            case 'logsnr_uniform':
                
                min_logsnr = -6
                max_logsnr = 2

                return lambda b: torch.sigmoid(-(torch.rand(b, device=self.device) * (max_logsnr - min_logsnr) + min_logsnr))
            case _:
                raise ValueError(f"Invalid noise distribution: {dist}")


class SelfForcingARCTrainingWrapper(ARCTrainingWrapper):
    '''
    ARC post-training with self-forcing: generates multi-chunk autoregressive rollouts
    using ping-pong sampling with randomized gradient attachment points, then scores
    the full rollout against real data using the ARC discriminator.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            discriminator: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            teacher_model: ConditionedDiffusionModelWrapper = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None,
            accomp_kwargs = None,
            context_router = False,
    ):
        super().__init__(
            model=model,
            discriminator=discriminator,
            arc_config=arc_config,
            optimizer_configs=optimizer_configs,
            teacher_model=teacher_model,
            use_ema=use_ema,
            pre_encoded=pre_encoded,
            cfg_dropout_prob=cfg_dropout_prob,
            timestep_sampler=timestep_sampler,
            validation_timesteps=validation_timesteps,
            clip_grad_norm=clip_grad_norm,
            trim_config=trim_config,
            inpainting_config=inpainting_config,
            accomp_kwargs=accomp_kwargs,
        )

        # Self-forcing config
        self_forcing_config = arc_config.get('self_forcing', {})
        self.max_sampling_steps = self_forcing_config.get('max_sampling_steps', 8)
        self.n_rollout_chunks = self_forcing_config.get('n_rollout_chunks', 4)
        self.chunk_size = self_forcing_config.get('chunk_size', 48)  # in latent frames
        self.context_size = self_forcing_config.get('context_size', 208)  # in latent frames
        self.use_kv_cache = self_forcing_config.get('use_kv_cache', False)
        self.window_size = self.context_size + self.chunk_size
        self.context_router = context_router

        # Gradient accumulation
        self.gradient_accumulation_steps = self_forcing_config.get('gradient_accumulation_steps', 1)
        self._micro_step = 0  # increments every training_step call
        self._optim_step = 0  # increments every actual optimizer.step()

        # Outpainting dropout probs for initial context
        self.outpainting_dropout_probs = self_forcing_config.get('outpainting_dropout_probs', None)
        self.use_context_shift = self_forcing_config.get('use_context_shift', True)

        # Block-causal streaming config
        if self.inpainting_config is not None:
            self.context_router_attention_pattern = self.inpainting_config.get("mask_kwargs", {}).get("context_router_attention_pattern", "enc-dec")
        else:
            self.context_router_attention_pattern = "enc-dec"
        self.n_window_chunks = self.context_size // self.chunk_size
        self.rope_mode = self_forcing_config.get('rope_mode', None)
        self.block_causal_bootstrap_mask = None  # lazy-init

        # Build noise schedule once: logsnr linspace [-6, 2] -> t = sigmoid(-logsnr)
        logsnr = torch.linspace(-6, 2, self.max_sampling_steps + 1)
        t_schedule = torch.sigmoid(-logsnr)
        t_schedule[0] = 1.0
        t_schedule[-1] = 0.0
        self.register_buffer('t_schedule', t_schedule)

        # Lazy-init FlexAttention block mask
        self.self_attention_block_mask = None

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            print(f"Inpainting mask kwargs: {self.inpaint_mask_kwargs}")

        # Teacher inpainting support detection
        if self.teacher_model is not None:
            self.teacher_has_inpainting = len(self.teacher_model.input_add_ids) > 0
        else:
            self.teacher_has_inpainting = True  # assume same architecture if no separate teacher
        # Allow config override
        self.teacher_has_inpainting = arc_config.get('teacher_has_inpainting', self.teacher_has_inpainting)
        self.student_objective = self.diffusion.diffusion_objective
        self.teacher_objective = self.teacher_model.diffusion_objective if self.teacher_model is not None else self.student_objective
        self.discriminator_objective = self.discriminator.diffusion_objective
        print(f"Student diffusion objective: {self.student_objective}")
        print(f"Teacher diffusion objective: {self.teacher_objective}")
        print(f"Discriminator diffusion objective: {self.discriminator_objective}")
        print(f"Teacher has inpainting support: {self.teacher_has_inpainting}")

    @staticmethod
    def _get_alphas_sigmas(t, objective):
        """Get (alpha, sigma) scaling factors for timestep t based on diffusion objective."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return 1 - t, t
        elif objective in ("v", "v_denoiser"):
            return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    @staticmethod
    def _noise_samples(x_0, noise, t, objective):
        """Create noised samples: x_t = alpha * x_0 + sigma * noise."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return x_0 * (1 - t)[:, None, None] + noise * t[:, None, None]
        elif objective in ("v", "v_denoiser"):
            alpha = torch.cos(t * math.pi / 2)[:, None, None]
            sigma = torch.sin(t * math.pi / 2)[:, None, None]
            return x_0 * alpha + noise * sigma
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    @staticmethod
    def _get_velocity_target(x_0, noise, t, objective):
        """Get the target velocity for a given diffusion objective."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return noise - x_0
        elif objective in ("v", "v_denoiser"):
            alpha = torch.cos(t * math.pi / 2)[:, None, None]
            sigma = torch.sin(t * math.pi / 2)[:, None, None]
            return noise * alpha - x_0 * sigma
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    @staticmethod
    def _get_x0_from_v_pred(x_t, v, t, objective):
        """Recover x_0 prediction from velocity prediction."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return x_t - t[:, None, None] * v
        elif objective in ("v", "v_denoiser"):
            alpha = torch.cos(t * math.pi / 2)[:, None, None]
            sigma = torch.sin(t * math.pi / 2)[:, None, None]
            return alpha * x_t - sigma * v
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    @staticmethod
    def _convert_sigma_to_t(sigma, objective):
        """Convert a noise level sigma to the model's native timestep."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return sigma  # RF: t = sigma
        elif objective in ("v", "v_denoiser"):
            return (2 / math.pi) * torch.arcsin(sigma.clamp(0, 1))
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    # ----------------------------------------------------------------
    # Scalar schedule helpers (for rollout inner loops)
    # ----------------------------------------------------------------

    @staticmethod
    def _alpha_for_sigma(sigma, objective):
        """Given a noise level sigma (scalar or tensor), return the corresponding alpha."""
        if objective in ("rectified_flow", "rf_denoiser"):
            return 1 - sigma
        elif objective in ("v", "v_denoiser"):
            return (1 - sigma ** 2).sqrt()
        else:
            raise ValueError(f"Unknown diffusion objective: {objective}")

    def _denoise_at_sigma(self, x_t, v, sigma):
        """Recover x_0 from model output at noise level sigma (scalar). Uses student objective."""
        alpha = self._alpha_for_sigma(sigma, self.student_objective)
        if self.student_objective in ("rectified_flow", "rf_denoiser"):
            return x_t - sigma * v
        else:  # v
            return alpha * x_t - sigma * v

    def _model_t_for_sigma(self, sigma):
        """Convert noise level sigma (scalar) to the student model's native timestep."""
        return self._convert_sigma_to_t(sigma, self.student_objective)
    
    def _get_block_mask(self, device):
        """Build the FlexAttention block mask (lazy, cached). Supports enc-dec and block-causal patterns."""
        if self.self_attention_block_mask is not None:
            return self.self_attention_block_mask

        fixed_mask_size = self.context_size
        seq_len = self.window_size

        if self.context_router_attention_pattern == "block-causal":
            chunk_size = self.chunk_size
            n_window_chunks = self.n_window_chunks
            n_total_chunks = n_window_chunks + 1

            def block_causal_mask(b, h, q_idx, kv_idx):
                q_block = (q_idx // chunk_size).clamp(max=n_total_chunks - 1)
                kv_block = (kv_idx // chunk_size).clamp(max=n_total_chunks - 1)
                return (kv_block <= q_block) & (kv_block >= q_block - n_window_chunks + 1)

            mask_mod = block_causal_mask
        else:
            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx < fixed_mask_size

            def postfix_mask(b, h, q_idx, kv_idx):
                return q_idx >= fixed_mask_size

            mask_mod = or_masks(prefix_mask, postfix_mask)

        # +1 for the postpend conditioning token
        self.self_attention_block_mask = create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=seq_len + 1, KV_LEN=seq_len + 1,
            device=device, _compile=True
        )
        print(f"Created self-attention block mask ({self.context_router_attention_pattern}) with context size {self.context_size}")
        print(self.self_attention_block_mask.to_string())
        return self.self_attention_block_mask

    def _apply_context_dropout(self, diffusion_input, accomp=None):
        """
        Prepare the input context region for ARC rollouts.

        Accompaniment regime (accomp is not None): every batch deterministically has
        zero context for the target and full unmodified accomp history, so the model
        always learns to bootstrap generation from past-accomp-only.

        Unconditional regime (accomp is None): retain stochastic context-dropout from
        `outpainting_dropout_probs` (uncond/partial drop, optional shift) so the model
        sees a mix of context lengths.

        Args:
            diffusion_input: (B, C, total_len) full latent sequence
            accomp: optional (B, C, total_len) accompaniment latents

        Returns:
            result: (B, C, total_len) modified target sequence
            (and accomp passed through unchanged when provided)
        """
        B, _, total_len = diffusion_input.shape

        if accomp is not None:
            result = diffusion_input.clone()
            ctx = self.context_size
            result[:, :, :ctx] = 0
            return result, accomp.clone()

        if self.outpainting_dropout_probs is None:
            return diffusion_input.clone()

        p_uncond = self.outpainting_dropout_probs.get('p_uncond', 0.0)
        p_partial = self.outpainting_dropout_probs.get('p_partial', 0.0)

        result = diffusion_input.clone()

        for i in range(B):
            rand_val = random.random()
            if rand_val < p_uncond:
                drop_length = self.context_size
            elif rand_val < p_uncond + p_partial and self.context_size > self.chunk_size:
                num_chunks = self.context_size // self.chunk_size
                chunks_to_drop = random.randint(1, num_chunks)
                drop_length = chunks_to_drop * self.chunk_size
            else:
                drop_length = 0

            if drop_length > 0:
                zeros = torch.zeros_like(result[i, :, :drop_length])
                if self.use_context_shift:
                    result[i] = torch.cat([zeros, diffusion_input[i, :, :total_len - drop_length]], dim=-1)
                else:
                    result[i, :, :drop_length] = zeros

        return result

    def _stack_accomp(self, metadata, total_len, dtype):
        """Stack per-sample accomp_input from metadata into a (B, C, total_len) tensor.

        Returns None when accomp_kwargs is unset. Pads/truncates each sample to total_len.
        """
        if self.accomp_kwargs is None:
            return None
        assert "accomp_input" in metadata[0], (
            "accomp_kwargs is set but no 'accomp_input' in metadata — set stem_mapping_path on the dataset."
        )
        per_item = []
        for md in metadata:
            a = md["accomp_input"]
            if a.shape[-1] < total_len:
                pad = torch.zeros(a.shape[0], total_len - a.shape[-1], dtype=a.dtype)
                a = torch.cat([a, pad], dim=-1)
            elif a.shape[-1] > total_len:
                a = a[..., :total_len]
            per_item.append(a)
        return torch.stack(per_item, dim=0).to(self.device, dtype=dtype)

    def _inject_accomp_window(self, conditioning, shifted_accomp, chunk_idx):
        """Slice the accomp window for the current rollout chunk, apply cutoff, and set conditioning.

        Window covers `[chunk_idx*chunk_size : chunk_idx*chunk_size + window_size)` of the shifted
        accomp. Cutoff is fixed across rollout chunks (latency buffer is preserved).
        """
        if self.accomp_kwargs is None or shifted_accomp is None:
            return
        start = chunk_idx * self.chunk_size
        accomp_window = shifted_accomp[:, :, start:start + self.window_size]
        # Deterministic cutoff during rollout — randomness already happened in base RF training.
        kwargs = {**self.accomp_kwargs, "p_uncond": 0.0, "p_partial": 0.0}
        accomp_masked, accomp_mask = apply_accomp_cutoff(accomp_window, **kwargs)
        conditioning['accomp_input'] = [accomp_masked]
        conditioning['accomp_mask'] = [accomp_mask]

    def _inject_disc_accomp(self, disc_conditioning, shifted_accomp):
        """Set the discriminator's accomp conditioning to the *full* rollout-length accomp (no cutoff).

        The discriminator scores the entire (context + rollout) sequence and benefits from the full
        accompaniment as conditioning context.
        """
        if self.accomp_kwargs is None or shifted_accomp is None:
            return
        B, _, L = shifted_accomp.shape
        accomp_mask = torch.ones(B, 1, L, device=shifted_accomp.device, dtype=shifted_accomp.dtype)
        disc_conditioning['accomp_input'] = [shifted_accomp]
        disc_conditioning['accomp_mask'] = [accomp_mask]

    def rf_warmup_step(self, diffusion_input, metadata, padding_masks):
        """Standard RF / v-diffusion supervised loss on real (windowed) data.

        Mirrors ode_warmup_step's inpainting/accomp setup, but skips the teacher
        rollout — student is trained directly against the real noised target.
        """
        # truncate to window size for RF warmup (ODE warmup is full sequence to avoid seq-len issues with teacher rollouts)
        diffusion_input = diffusion_input[:, :, :self.window_size]
        conditioning = self.diffusion.conditioner(metadata, self.device)

        inpaint_mask = None
        inpaint_masked_input = None
        context_router = None
        block_mask = None

        if self.inpainting_config is not None:
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(
                diffusion_input, padding_masks=padding_masks,
                **self.inpaint_mask_kwargs,
            )
            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]
            if hasattr(self, "_get_block_mask"):
                block_mask = self._get_block_mask(diffusion_input.device)
                context_router = 1 - inpaint_mask

        if self.accomp_kwargs is not None:
            accomp_full = self._stack_accomp(metadata, diffusion_input.shape[-1], diffusion_input.dtype)
            kwargs = {**self.accomp_kwargs, "p_uncond": 0.0, "p_partial": 0.0}
            accomp_input, accomp_mask = apply_accomp_cutoff(accomp_full, **kwargs)
            conditioning['accomp_input'] = [accomp_input]
            conditioning['accomp_mask'] = [accomp_mask]

        t = self.dis_noise_dist(diffusion_input.shape[0])
        noise = torch.randn_like(diffusion_input)
        x_t = self._noise_samples(diffusion_input, noise, t, self.student_objective)
        target_v = self._get_velocity_target(diffusion_input, noise, t, self.student_objective)

        v_pred = self.diffusion(
            x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob,
            context_router_mask=context_router, self_attention_block_mask=block_mask,
        )

        rf_loss = F.mse_loss(v_pred, target_v, reduction='none')
        if self.inpainting_config is not None:
            loss_mask = (1 - inpaint_mask).bool()
            if loss_mask.ndim == 2 and rf_loss.ndim == 3:
                loss_mask = loss_mask.unsqueeze(1)
            if loss_mask.shape[1] != rf_loss.shape[1]:
                loss_mask = loss_mask.repeat(1, rf_loss.shape[1], 1)
            rf_loss = rf_loss[loss_mask].mean()
        else:
            rf_loss = rf_loss.mean()
        return rf_loss

    def ode_warmup_step(self, diffusion_input, metadata, padding_masks):
        """
        Unified ODE warmup supporting:
        - RF or v-diffusion teacher (picks correct sampler)
        - RF or v-diffusion student (objective-aware denoising)
        - Teacher with or without inpainting (derives context from teacher output if needed)
        - noise_level vs timestep alignment between teacher and student
        """
        refresh_diff_states = self.global_step % self.ode_refresh_rate == 0

        if refresh_diff_states:
            start_noise = torch.randn_like(diffusion_input)
            teacher_conditioning = self.teacher_model.conditioner(metadata, self.device)

            teacher_inpaint_masked_input = None
            teacher_inpaint_mask = None

            # Set up inpainting for teacher (only if teacher supports it)
            if self.inpainting_config is not None and self.teacher_has_inpainting:
                teacher_inpaint_masked_input, teacher_inpaint_mask = random_inpaint_mask(
                    diffusion_input, padding_masks=padding_masks,
                    **self.inpaint_mask_kwargs,
                )
                teacher_conditioning['inpaint_mask'] = [teacher_inpaint_mask]
                teacher_conditioning['inpaint_masked_input'] = [teacher_inpaint_masked_input]
                self.ode_inpaint_mask = teacher_inpaint_mask
                self.ode_inpaint_masked_input = teacher_inpaint_masked_input

                if hasattr(self, "_get_block_mask"):
                    self.ode_block_mask = self._get_block_mask(diffusion_input.device)
                    self.ode_context_router = 1 - teacher_inpaint_mask

            # Set up accompaniment conditioning for the warmup window. Accomp is the matching
            # slice of the paired stem with the configured cutoff applied. Cached on self so the
            # student-training branch below uses the same accomp.
            self.ode_accomp_input = None
            self.ode_accomp_mask = None
            if self.accomp_kwargs is not None:
                accomp_full = self._stack_accomp(metadata, diffusion_input.shape[-1], diffusion_input.dtype)
                kwargs = {**self.accomp_kwargs, "p_uncond": 0.0, "p_partial": 0.0}
                self.ode_accomp_input, self.ode_accomp_mask = apply_accomp_cutoff(accomp_full, **kwargs)
                teacher_conditioning['accomp_input'] = [self.ode_accomp_input]
                teacher_conditioning['accomp_mask'] = [self.ode_accomp_mask]

            # Create timestep schedule based on teacher's objective
            logsnr = torch.linspace(-6, 2, self.ode_n_sampling_steps + 1)
            sigma = torch.sigmoid(-logsnr)
            sigma[0] = 1.0
            sigma[-1] = 0.0

            if self.teacher_objective in ("rectified_flow", "rf_denoiser"):
                t_teacher_schedule = sigma  # RF: t = sigma
                sampler = sample_flow_dpmpp_w_intermediates
            else:  # v-diffusion
                t_teacher_schedule = torch.linspace(1, 0, self.ode_n_sampling_steps + 1)
                t_teacher_schedule[0] = 1.0
                t_teacher_schedule[-1] = 0.0
                sampler = sample_v_dpmpp_w_intermediates

            self.ode_metadata = metadata
            self.diff_states = sampler(
                self.teacher_model, start_noise, sigmas=t_teacher_schedule,
                cond=teacher_conditioning,
                cfg_scale=self.ode_warmup_cfg, dist_shift=self.teacher_model.dist_shift, batch_cfg=True,
                inpaint_masked_input=teacher_inpaint_masked_input,
                inpaint_mask=teacher_inpaint_mask,
                context_router_mask=self.ode_context_router if self.inpainting_config is not None and self.teacher_has_inpainting and hasattr(self, "_get_block_mask") else None,
                self_attention_block_mask=self.ode_block_mask if self.inpainting_config is not None and self.teacher_has_inpainting and hasattr(self, "_get_block_mask") else None
            )

            # If teacher doesn't support inpainting but student does,
            # derive inpaint context from teacher's generated output
            if self.inpainting_config is not None and not self.teacher_has_inpainting:
                self.diff_states['target'] = self.diff_states['target'][..., :self.window_size]  # 
                teacher_target = self.diff_states['target']  # (B, C, T)
                # truncate intermediates to window size in case teacher generates longer sequences than student
                self.diff_states['x'] = self.diff_states['x'][..., :self.window_size]
                B_t, C_t, T_t = teacher_target.shape
                self.ode_inpaint_mask = torch.zeros(B_t, 1, T_t, device=diffusion_input.device)
                self.ode_inpaint_mask[:, :, :self.context_size] = 1.0
                self.ode_inpaint_masked_input = torch.zeros_like(teacher_target)
                self.ode_inpaint_masked_input[:, :, :self.context_size] = teacher_target[:, :, :self.context_size]

                if hasattr(self, "_get_block_mask"):
                    self.ode_block_mask = self._get_block_mask(diffusion_input.device)
                    self.ode_context_router = 1 - self.ode_inpaint_mask

        # --- Train student on teacher's ODE intermediates ---
        conditioning = self.diffusion.conditioner(self.ode_metadata, self.device)

        if self.inpainting_config is not None:
            conditioning['inpaint_mask'] = [self.ode_inpaint_mask]
            conditioning['inpaint_masked_input'] = [self.ode_inpaint_masked_input]

        if getattr(self, 'ode_accomp_input', None) is not None:
            conditioning['accomp_input'] = [self.ode_accomp_input]
            conditioning['accomp_mask'] = [self.ode_accomp_mask]

        ixs = torch.randint(0, self.ode_n_sampling_steps, (diffusion_input.shape[0],))
        t_teacher = self.diff_states['t'].clone().detach()[ixs].squeeze(-1)
        x_t = self.diff_states['x'].clone().detach()[ixs.flatten(), torch.arange(diffusion_input.shape[0])].squeeze(0)

        # Convert teacher timestep to student timestep based on alignment mode
        if getattr(self, 'ode_noise_alignment', 'noise_level') == "noise_level":
            # Same noise level: get sigma from teacher's t, then find student's native t
            _, sigma = self._get_alphas_sigmas(t_teacher, self.teacher_objective)
            t_student = self._convert_sigma_to_t(sigma, self.student_objective)
        else:  # "timestep"
            # Same timestep value: pass teacher's t directly to student
            t_student = t_teacher

        t_student = t_student.to(self.device).detach().clone().requires_grad_(False)
        x_t = x_t.to(self.device).detach().clone().requires_grad_(False)

        v_t_student = self.diffusion(x_t, t_student, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob,
            context_router_mask=self.ode_context_router if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None,
            self_attention_block_mask=self.ode_block_mask if self.inpainting_config is not None and hasattr(self, "_get_block_mask") else None
        )

        # Denoise using student's own objective
        denoised_student = self._get_x0_from_v_pred(x_t, v_t_student, t_student, self.student_objective)
        if self.inpainting_config is not None:
            denoised_student = denoised_student * (1 - self.ode_inpaint_mask) + self.ode_inpaint_masked_input * self.ode_inpaint_mask
        ode_mse_loss = F.mse_loss(denoised_student, self.diff_states['target'], reduction='none')
        # if there's a mask, apply it to the loss
        if self.inpainting_config is not None:
            loss_mask = (1 - self.ode_inpaint_mask).bool()
            if loss_mask.ndim == 2 and ode_mse_loss.ndim == 3:
                loss_mask = loss_mask.unsqueeze(1)

            if loss_mask.shape[1] != ode_mse_loss.shape[1]:
                loss_mask = loss_mask.repeat(1, ode_mse_loss.shape[1], 1)
            ode_mse_loss = ode_mse_loss[loss_mask].mean() 
        else:
            ode_mse_loss = ode_mse_loss.mean()

        return ode_mse_loss


    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=1,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def _build_model_kwargs(self, context_router_mask, block_mask, kv_cache, step_i, chunk_idx,
                             streaming_cache=None, use_streaming=False):
        """Build per-step kwargs for diffusion model call.

        Returns a dict of extra kwargs to pass to self.diffusion().
        Handles enc-dec and block-causal streaming (fused first step) paths.
        """
        kwargs = {
            'context_router_mask': context_router_mask,
        }

        if use_streaming:
            # Block-causal streaming: cache persists across chunks
            kwargs['use_kv_cache'] = True
            kwargs['kv_cache'] = kv_cache
            if step_i == 0:
                if chunk_idx == 0:
                    # Bootstrap: full input with block-causal mask
                    kwargs['self_attention_block_mask'] = block_mask
                else:
                    # Fused first step: pending + target with streaming mask
                    kwargs.update(streaming_cache.get_first_step_kwargs(context_router_mask.device))
            # steps 1+: standard cached path (no block mask needed)

            # RoPE positions — fused first step spans an extra chunk (the about-to-be-ejected one).
            is_fused_first_step = (step_i == 0 and chunk_idx > 0)
            rope_positions = streaming_cache.get_rope_positions(
                context_router_mask.shape[0], context_router_mask.device, first_step=is_fused_first_step
            )
            if rope_positions is not None:
                kwargs['per_item_rope_positions'] = rope_positions
        elif self.use_kv_cache:
            # Enc-dec KV cache: reset each chunk, prefill handles first pass
            kwargs['self_attention_block_mask'] = None
            kwargs['kv_cache'] = kv_cache
            kwargs['use_kv_cache'] = True
            kwargs['prefill'] = True
        else:
            # No KV cache: always pass block mask
            kwargs['self_attention_block_mask'] = block_mask

        return kwargs

    def generate_rollout(self, diffusion_input, conditioning, padding_masks):
        """
        Generate a multi-chunk autoregressive rollout using ping-pong sampling
        with self-forcing gradient flow.

        Args:
            diffusion_input: Full latent sequence (B, C, context_size + chunk_size * n_rollout_chunks)
            conditioning: Dict of conditioning tensors (text, etc.)
            padding_masks: (B, seq_len) padding masks

        Returns:
            full_student: (B, C, context_size + chunk_size * n_rollout_chunks) — context + generated rollout
            shifted_input: (B, C, same) — the shifted real sequence (for discriminator)
        """
        B, C, total_len = diffusion_input.shape
        device = diffusion_input.device

        # Stack and shift accompaniment in lockstep with target. Metadata is stashed by
        # training_step on `self._current_metadata` since generate_rollout's signature doesn't
        # carry metadata.
        accomp_full = self._stack_accomp(self._current_metadata, total_len, diffusion_input.dtype) \
            if (self.accomp_kwargs is not None and getattr(self, '_current_metadata', None) is not None) else None

        if accomp_full is not None:
            shifted_input, shifted_accomp = self._apply_context_dropout(diffusion_input, accomp=accomp_full)
        else:
            shifted_input = self._apply_context_dropout(diffusion_input)
            shifted_accomp = None

        # Stash for the discriminator path in training_step.
        self._last_shifted_accomp = shifted_accomp

        # Extract context from the shifted input
        context = shifted_input[:, :, :self.context_size].clone()

        # Initialize inpaint_input: [context | zeros]
        inpaint_input = torch.zeros(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)
        inpaint_input[:, :, :self.context_size] = context

        # Inpainting mask: 1 = keep (context), 0 = generate (new chunk)
        mask = torch.zeros(B, 1, self.window_size, device=device, dtype=diffusion_input.dtype)
        mask[:, :, :self.context_size] = 1.0

        # context_router_mask marks the decoder region (inverted mask)
        context_router_mask = 1.0 - mask

        # Get FlexAttention block mask — eagerly before any compiled region
        block_mask = self._get_block_mask(device)

        rollout_chunks = []

        # Determine if we use block-causal streaming
        use_streaming = (self.context_router_attention_pattern == "block-causal" and self.use_kv_cache)

        if use_streaming:
            streaming_cache = BlockCausalStreamingCache(
                self.diffusion.model, self.chunk_size, self.n_window_chunks,
                rope_mode=self.rope_mode,
                accomp_cutoff=(self.accomp_kwargs or {}).get('cutoff'),
            )
            kv_cache = streaming_cache.init_kv_cache()
            # Eagerly create the first-step mask so create_block_mask doesn't run
            # inside a torch.compile'd forward pass (nested compilation → CUDA errors)
            streaming_cache.get_first_step_mask(device)
        elif self.use_kv_cache:
            streaming_cache = None
            kv_cache = None  # will be initialized per-chunk
        else:
            streaming_cache = None
            kv_cache = None

        # pregenerate all k's to make it sync across gpus
        k_list = self.generate_and_sync_list(self.n_rollout_chunks, self.max_sampling_steps, device)

        for chunk_idx in range(self.n_rollout_chunks):
            # Sample number of steps for this chunk
            k = k_list[chunk_idx]

            # Fresh noise for the full window
            x = torch.randn(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)

            # Update conditioning with current inpaint input/mask
            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [inpaint_input]
            self._inject_accomp_window(conditioning, shifted_accomp, chunk_idx)

            # Per-chunk KV cache reset for enc-dec (not streaming)
            if self.use_kv_cache and not use_streaming:
                self.diffusion.model.model.transformer.clear_kv_cache()
                kv_cache = {
                    'initialized': False,
                }

            # Timestep as batch tensor
            ts = torch.ones(B, device=device, dtype=diffusion_input.dtype)

            for step_i in range(k):
                t_curr = self.t_schedule[step_i]
                t_next = self.t_schedule[step_i + 1]

                # Apply inpainting: noise the context to match current noise level
                noised_inpaint = inpaint_input * (1 - t_curr) + torch.randn_like(x) * t_curr
                x = x * (1 - mask) + noised_inpaint * mask

                t_batch = t_curr * ts

                step_kv_cache = kv_cache

                step_kwargs = self._build_model_kwargs(
                    context_router_mask, block_mask, step_kv_cache, step_i, chunk_idx,
                    streaming_cache=streaming_cache, use_streaming=use_streaming,
                )

                if step_i < k - 1:
                    # No-grad steps
                    with torch.no_grad():
                        v = self.diffusion(
                            x, t_batch, cond=conditioning,
                            cfg_dropout_prob=self.cfg_dropout_prob,
                            **step_kwargs
                        )
                        if step_i == 0 and chunk_idx > 0 and use_streaming:
                            streaming_cache.finalize_first_step()
                        denoised = x - t_curr * v
                        # Ping-pong: re-noise to next level
                        x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
                else:
                    # Gradient step: detach input, run with gradients
                    v = self.diffusion(
                        x, t_batch, cond=conditioning,
                        cfg_dropout_prob=self.cfg_dropout_prob,
                        **step_kwargs
                    )
                    if step_i == 0 and chunk_idx > 0 and use_streaming:
                        streaming_cache.finalize_first_step()
                    denoised = x - t_curr * v
                    x = denoised  # final clean output

            # Apply final inpainting (clean context in the encoder region)
            x = x * (1 - mask) + inpaint_input * mask

            # Extract generated chunk (the decoder/rightmost region)
            chunk = x[:, :, -self.chunk_size:]
            rollout_chunks.append(chunk)

            # Slide window for next chunk
            with torch.no_grad():
                if use_streaming:
                    # Advance streaming cache: eject oldest, set pending
                    streaming_cache.advance(chunk.detach())

                # Slide inpaint_input: shift left by chunk_size, append zeros
                inpaint_input = inpaint_input.detach()
                inpaint_input[:, :, -self.chunk_size:] = chunk.detach()
                inpaint_input = torch.cat([
                    inpaint_input[:, :, self.chunk_size:],
                    torch.zeros(B, C, self.chunk_size, device=device, dtype=diffusion_input.dtype)
                ], dim=2)

        # Clear KV cache after rollout to free GPU memory
        if self.use_kv_cache:
            self.diffusion.model.model.transformer.clear_kv_cache()

        # Concatenate context + generated chunks for the full student sequence
        full_student = torch.cat([context, torch.cat(rollout_chunks, dim=2)], dim=2)
        return full_student, shifted_input

    @torch.no_grad()
    def generate_demo_rollout(self, diffusion_input, conditioning, demo_steps=None, n_chunks=None,
                              time_varying_prompts: tp.Optional[tp.Dict[int, tp.Union[str, tp.List[tp.Tuple[str, float]]]]] = None,
                              prompt_cond_key: str = "prompt",
                              eject_buffer_n: int = 0):
        """
        Generate a multi-chunk autoregressive rollout for demo/inference.
        Unlike generate_rollout, this runs fully no-grad with all steps per chunk
        and no self-forcing randomness.

        Args:
            diffusion_input: (B, C, L) latent input — only the first context_size frames are used as initial context
            conditioning: Dict of conditioning tensors (from conditioner)
            demo_steps: Number of sampling steps per chunk (defaults to max_sampling_steps)
            n_chunks: Number of chunks to generate (defaults to n_rollout_chunks)

        Returns:
            context: (B, C, context_size) — the initial context used
            rollout: (B, C, chunk_size * n_chunks) — concatenated generated chunks
        """
        if demo_steps is None:
            demo_steps = self.max_sampling_steps
        if n_chunks is None:
            n_chunks = self.n_rollout_chunks

        B, C, _ = diffusion_input.shape
        device = diffusion_input.device

        # Build the t_schedule for the requested number of steps
        logsnr = torch.linspace(-6, 2, demo_steps + 1, device=device)
        t_schedule = torch.sigmoid(-logsnr)
        t_schedule[0] = 1.0
        t_schedule[-1] = 0.0

        # Apply context dropout via prepend-and-shift (same as training)
        total_len = diffusion_input.shape[-1]
        accomp_full = self._stack_accomp(self._current_metadata, total_len, diffusion_input.dtype) \
            if (self.accomp_kwargs is not None and getattr(self, '_current_metadata', None) is not None) else None
        if accomp_full is not None:
            shifted_input, shifted_accomp = self._apply_context_dropout(diffusion_input, accomp=accomp_full)
        else:
            shifted_input = self._apply_context_dropout(diffusion_input)
            shifted_accomp = None
        self._last_shifted_accomp = shifted_accomp

        # Extract context from the shifted input
        context = shifted_input[:, :, :self.context_size].clone()

        # Initialize inpaint_input: [context | zeros]
        inpaint_input = torch.zeros(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)
        inpaint_input[:, :, :self.context_size] = context

        # Inpainting mask: 1 = keep (context), 0 = generate
        mask = torch.zeros(B, 1, self.window_size, device=device, dtype=diffusion_input.dtype)
        mask[:, :, :self.context_size] = 1.0

        context_router_mask = 1.0 - mask
        block_mask = self._get_block_mask(device)

        rollout_chunks = []

        model = self.diffusion_ema.ema_model if self.diffusion_ema is not None else self.diffusion.model

        # Determine if we use block-causal streaming
        use_streaming = (self.context_router_attention_pattern == "block-causal" and self.use_kv_cache)

        if use_streaming:
            streaming_cache = BlockCausalStreamingCache(
                model, self.chunk_size, self.n_window_chunks,
                rope_mode=self.rope_mode,
                accomp_cutoff=(self.accomp_kwargs or {}).get('cutoff'),
            )
            kv_cache = streaming_cache.init_kv_cache()
            streaming_cache.get_first_step_mask(device)
        elif self.use_kv_cache:
            streaming_cache = None
            kv_cache = None
        else:
            streaming_cache = None
            kv_cache = None

        if time_varying_prompts is not None:
            t5 = self.diffusion.conditioner.conditioners[prompt_cond_key]
            assert isinstance(t5, T5Conditioner), (
                f"time_varying_prompts requires conditioner '{prompt_cond_key}' to be a "
                f"T5Conditioner (or subclass), got {type(t5).__name__}"
            )

        prev_dominant = _dominant_prompt(time_varying_prompts[0]) if time_varying_prompts is not None else None

        for chunk_idx in range(n_chunks):
            x = torch.randn(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [inpaint_input]
            self._inject_accomp_window(conditioning, shifted_accomp, chunk_idx)

            is_rebootstrap = False
            if time_varying_prompts is not None:
                if chunk_idx not in time_varying_prompts:
                    raise ValueError(f"time_varying_prompts missing entry for chunk {chunk_idx}")
                entry = time_varying_prompts[chunk_idx]
                weighted = [(entry, 1.0)] if isinstance(entry, str) else [(str(p), float(w)) for p, w in entry]
                emb, msk = mix_t5_prompts(t5, [weighted for _ in range(B)], device)
                conditioning[prompt_cond_key] = [emb, msk]

                current_dominant = _dominant_prompt(entry)
                if chunk_idx > 0 and current_dominant != prev_dominant:
                    is_rebootstrap = True
                    print(f"[rebootstrap] chunk {chunk_idx}: dominant '{prev_dominant}' -> '{current_dominant}', re-priming KV cache")
                    if use_streaming:
                        kv_cache = streaming_cache.init_kv_cache()
                    if eject_buffer_n > 0:
                        n_zero = min(eject_buffer_n, self.n_window_chunks) * self.chunk_size
                        print(f"[rebootstrap] zeroing {n_zero} oldest context frames (eject_buffer_n={eject_buffer_n})")
                        inpaint_input[..., :n_zero] = 0
                        conditioning['inpaint_masked_input'] = [inpaint_input]
                prev_dominant = current_dominant

            cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)

            # Per-chunk KV cache reset for enc-dec (not streaming)
            if self.use_kv_cache and not use_streaming:
                self.diffusion.model.model.transformer.clear_kv_cache()
                kv_cache = {'initialized': False}

            ts = torch.ones(B, device=device, dtype=diffusion_input.dtype)

            for step_i in range(demo_steps):
                t_curr = t_schedule[step_i]
                t_next = t_schedule[step_i + 1]

                # Apply inpainting: noise the context to match current noise level
                noised_inpaint = inpaint_input * (1 - t_curr) + torch.randn_like(x) * t_curr
                x = x * (1 - mask) + noised_inpaint * mask

                t_batch = t_curr * ts

                # Build per-step kwargs
                step_kwargs = {'context_router_mask': context_router_mask}
                use_bootstrap = (chunk_idx == 0) or is_rebootstrap
                if time_varying_prompts is not None:
                    step_kwargs['cross_attn_no_cache'] = True
                if use_streaming:
                    step_kwargs['use_kv_cache'] = True
                    step_kwargs['kv_cache'] = kv_cache
                    if step_i == 0:
                        if use_bootstrap:
                            step_kwargs['self_attention_block_mask'] = block_mask
                        else:
                            step_kwargs.update(streaming_cache.get_first_step_kwargs(device))
                    is_fused_first_step = (step_i == 0 and not use_bootstrap)
                    rope_positions = streaming_cache.get_rope_positions(B, device, first_step=is_fused_first_step)
                    if rope_positions is not None:
                        step_kwargs['per_item_rope_positions'] = rope_positions
                elif self.use_kv_cache:
                    step_kwargs['self_attention_block_mask'] = block_mask if step_i == 0 else None
                    step_kwargs['kv_cache'] = kv_cache
                    step_kwargs['use_kv_cache'] = True
                    step_kwargs['prefill'] = True
                else:
                    step_kwargs['self_attention_block_mask'] = block_mask

                v = model(
                    x, t_batch,
                    **cond_inputs,
                    **step_kwargs
                )

                if step_i == 0 and not use_bootstrap and use_streaming:
                    streaming_cache.finalize_first_step()

                denoised = x - t_curr * v

                if step_i < demo_steps - 1:
                    # Ping-pong: re-noise to next level
                    x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
                else:
                    x = denoised

            # Apply final inpainting
            x = x * (1 - mask) + inpaint_input * mask

            chunk = x[:, :, -self.chunk_size:]
            rollout_chunks.append(chunk)

            # Slide window
            if use_streaming:
                streaming_cache.advance(chunk)

            inpaint_input[:, :, -self.chunk_size:] = chunk
            inpaint_input = torch.cat([
                inpaint_input[:, :, self.chunk_size:],
                torch.zeros(B, C, self.chunk_size, device=device, dtype=diffusion_input.dtype)
            ], dim=2)

        # Clear KV cache after rollout to free GPU memory
        if self.use_kv_cache:
            self.diffusion.model.model.transformer.clear_kv_cache()

        rollout = torch.cat(rollout_chunks, dim=2)
        return context, rollout
    
    @torch.no_grad()
    def generate_teacher_demo(self, diffusion_input, metadata, sampling_steps=None):
        """
        Run the teacher model sampler (same as ode_warmup_step) and return the
        denoised target. Used as a debug tool to verify teacher samples are meaningful.
        """
        # truncate input to window size
        diffusion_input = diffusion_input[..., :self.window_size]
        if self.teacher_model is None:
            return None

        if sampling_steps is None:
            sampling_steps = self.ode_n_sampling_steps

        B, C, T = diffusion_input.shape
        start_noise = torch.randn_like(diffusion_input)
        teacher_conditioning = self.teacher_model.conditioner(metadata, self.device)

        teacher_inpaint_masked_input = None
        teacher_inpaint_mask = None

        # Set up inpainting context for teacher if it supports it
        if self.inpainting_config is not None and self.teacher_has_inpainting:
            # Use a simple context mask: first context_size frames are context
            print("Generating teacher demo with inpainting context (first", self.context_size, "frames)")
            teacher_inpaint_mask = torch.zeros(B, 1, T, device=self.device)
            teacher_inpaint_mask[:, :, :self.context_size] = 1.0
            teacher_inpaint_masked_input = torch.zeros_like(diffusion_input)
            teacher_inpaint_masked_input[:, :, :self.context_size] = diffusion_input[:, :, :self.context_size]
            teacher_conditioning['inpaint_mask'] = [teacher_inpaint_mask]
            teacher_conditioning['inpaint_masked_input'] = [teacher_inpaint_masked_input]

        # Inject accompaniment for the teacher window if configured. Required for the teacher
        # whenever it declares accomp in its input_add_ids — otherwise input_add channel count
        # mismatches `to_input_add_embed`'s expected input dim.
        if self.accomp_kwargs is not None:
            accomp_full = self._stack_accomp(metadata, T, diffusion_input.dtype)
            kwargs = {**self.accomp_kwargs, "p_uncond": 0.0, "p_partial": 0.0}
            accomp_masked, accomp_mask = apply_accomp_cutoff(accomp_full, **kwargs)
            teacher_conditioning['accomp_input'] = [accomp_masked]
            teacher_conditioning['accomp_mask'] = [accomp_mask]

        # Build timestep schedule based on teacher's objective
        logsnr = torch.linspace(-6, 2, sampling_steps + 1)
        sigma = torch.sigmoid(-logsnr)
        sigma[0] = 1.0
        sigma[-1] = 0.0

        if self.teacher_objective in ("rectified_flow", "rf_denoiser"):
            t_schedule = sigma
            sampler = sample_flow_dpmpp_w_intermediates
        else:
            t_schedule = torch.linspace(1, 0, sampling_steps + 1)
            t_schedule[0] = 1.0
            t_schedule[-1] = 0.0
            sampler = sample_v_dpmpp_w_intermediates

        context_router_mask = (1 - teacher_inpaint_mask) if teacher_inpaint_mask is not None and hasattr(self, "_get_block_mask") else None
        block_mask = self._get_block_mask(self.device) if context_router_mask is not None else None

        diff_states = sampler(
            self.teacher_model, start_noise, sigmas=t_schedule,
            cond=teacher_conditioning,
            cfg_scale=self.ode_warmup_cfg, dist_shift=self.teacher_model.dist_shift, batch_cfg=True,
            inpaint_masked_input=teacher_inpaint_masked_input,
            inpaint_mask=teacher_inpaint_mask,
            context_router_mask=context_router_mask,
            self_attention_block_mask=block_mask
        )

        return diff_states['target']

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        # Check for wrapped padding masks to avoid interpolation error
        first_padding_mask = metadata[0]["padding_mask"]
        if isinstance(first_padding_mask, list) and len(first_padding_mask) == 1:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
        else:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = F.interpolate(
                        padding_masks.unsqueeze(1).float(),
                        size=diffusion_input.shape[2], mode="nearest"
                    ).squeeze(1).bool()
            else:
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        grad_accum = self.gradient_accumulation_steps
        is_accum_start = (self._micro_step % grad_accum) == 0
        is_accum_end = ((self._micro_step + 1) % grad_accum) == 0

        if self._optim_step < self.ode_warmup_steps:
            diffusion_input = diffusion_input[..., :self.context_size + self.chunk_size]
            ode_mse_loss = self.ode_warmup_step(diffusion_input, metadata, padding_masks[..., :self.context_size + self.chunk_size])

            if is_accum_start:
                opt_gen.zero_grad()
            self.manual_backward(ode_mse_loss / grad_accum)

            if is_accum_end:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
                opt_gen.step()
                self._optim_step += 1

                if sched_gen is not None:
                    sched_gen.step()

                if self.diffusion_ema is not None:
                    self.diffusion_ema.update()

            self._micro_step += 1

            log_dict = {'train/ode_mse_loss': ode_mse_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            return ode_mse_loss.detach()

        if self._optim_step < self.rf_warmup_steps:
            diffusion_input = diffusion_input[..., :self.context_size + self.chunk_size]
            rf_loss = self.rf_warmup_step(diffusion_input, metadata, padding_masks[..., :self.context_size + self.chunk_size])

            if is_accum_start:
                opt_gen.zero_grad()
            self.manual_backward(rf_loss / grad_accum)

            if is_accum_end:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
                opt_gen.step()
                self._optim_step += 1

                if sched_gen is not None:
                    sched_gen.step()

                if self.diffusion_ema is not None:
                    self.diffusion_ema.update()

            self._micro_step += 1

            log_dict = {'train/rf_warmup_loss': rf_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            return rf_loss.detach()

        conditioning = self.diffusion.conditioner(metadata, self.device)

        # Use _optim_step so all micro-batches in an accumulation window stay in the same phase
        train_gen = (self._optim_step % 2) == 0

        # Stash metadata so generate_rollout can derive accompaniment latents per item.
        self._current_metadata = metadata

        # Generate rollout (the "denoised_student") and get shifted real for discriminator
        if train_gen:
            denoised_student, real_rollout = self.generate_rollout(diffusion_input, conditioning, padding_masks)
        else:
            diffusion_input.requires_grad_(False)
            with torch.no_grad():
                denoised_student, real_rollout = self.generate_rollout(diffusion_input, conditioning, padding_masks)
                denoised_student = denoised_student.detach()

        shifted_accomp = getattr(self, '_last_shifted_accomp', None)
        self._current_metadata = None

        # Clear inpainting tensors added to conditioning dict during rollout to free GPU memory
        conditioning.pop('inpaint_mask', None)
        conditioning.pop('inpaint_masked_input', None)
        conditioning.pop('accomp_input', None)
        conditioning.pop('accomp_mask', None)
        del conditioning

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            # Discriminator scoring for adversarial loss
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            x_t_gan = denoised_student * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]

            disc_conditioning = self.discriminator.conditioner(metadata, self.device)
            self._inject_disc_accomp(disc_conditioning, shifted_accomp)
        # with torch.compiler.disable():
            v_t_gan_hidden_states = self.discriminator(
                x_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, use_checkpointing=False,
                exit_layer_ix=self.discriminator_dit_layer
            )
            # truncate to the rollout region for scoring
            v_t_gan_hidden_states = v_t_gan_hidden_states[:, self.context_size:, :]
            disc_scores = self.discriminator_head(v_t_gan_hidden_states.transpose(1, 2))

            log_dict['gen_disc_scores_mean'] = disc_scores.mean().detach()
            log_dict['gen_disc_scores_std'] = disc_scores.std().detach()

            # Generator adversarial loss (relativistic variants need real scores too)
            if self.discriminator_loss_type in ('relativistic', 'relativistic-lsgan'):
                x_t_gan_real = real_rollout * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]
                v_t_gan_real_hidden_states = self.discriminator(
                    x_t_gan_real, t_gan, cond=disc_conditioning,
                    cfg_scale=1.0, use_checkpointing=False,
                    exit_layer_ix=self.discriminator_dit_layer
                )
                v_t_gan_real_hidden_states = v_t_gan_real_hidden_states[:, self.context_size:, :]
                disc_scores_real = self.discriminator_head(v_t_gan_real_hidden_states.transpose(1, 2))
            else:
                disc_scores_real = None

            loss_adv = self.calculate_gen_adv_loss(disc_scores_real, disc_scores)

            loss = loss_adv
            log_dict['train/adv_loss'] = loss_adv.detach()

            if is_accum_start:
                opt_gen.zero_grad()
            self.manual_backward(loss / grad_accum)

            if is_accum_end:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
                opt_gen.step()
                self._optim_step += 1

                if sched_gen is not None:
                    sched_gen.step()

                if self.diffusion_ema is not None:
                    self.diffusion_ema.update()

            log_dict['train/gen_loss'] = loss.detach()

        else:
            # Discriminator training step
            denoised_student = denoised_student.detach()
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            reals_t_gan = real_rollout * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]
            denoised_t_gan = denoised_student * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]

            reals_t_gan = reals_t_gan.detach()
            denoised_t_gan = denoised_t_gan.detach()

            disc_conditioning = self.discriminator.conditioner(metadata, self.device)
            self._inject_disc_accomp(disc_conditioning, shifted_accomp)
            # with torch.compiler.disable():
            reals_gan_hidden_states = self.discriminator(
                reals_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False,
            ).transpose(1, 2)
            reals_gan_hidden_states = reals_gan_hidden_states[:, :,  self.context_size:]
            denoised_gan_hidden_states = self.discriminator(
                denoised_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False,
            ).transpose(1, 2)
            denoised_gan_hidden_states = denoised_gan_hidden_states[:, :,  self.context_size:]
            disc_scores_reals = checkpoint(self.discriminator_head, reals_gan_hidden_states)
            disc_scores_denoised = checkpoint(self.discriminator_head, denoised_gan_hidden_states)

            if self.include_grad_penalties:
                r1_approx_variance = 0.05
                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                noised_reals_gan_hidden_states = self.discriminator(
                    noised_reals_t_gan, t_gan, cond=disc_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False
                ).transpose(1, 2)
                noised_denoised_gan_hidden_states = self.discriminator(
                    noised_denoised_t_gan, t_gan, cond=disc_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False
                ).transpose(1, 2)
                noised_reals_gan_hidden_states = noised_reals_gan_hidden_states[:, :, self.context_size:]
                noised_denoised_gan_hidden_states = noised_denoised_gan_hidden_states[:, :, self.context_size:]
                disc_scores_noised_reals = checkpoint(self.discriminator_head, noised_reals_gan_hidden_states)
                disc_scores_noised_denoised = checkpoint(self.discriminator_head, noised_denoised_gan_hidden_states)
                r1_diff = disc_scores_noised_reals - disc_scores_reals
                r2_diff = disc_scores_noised_denoised - disc_scores_denoised
                r1_penalty = torch.sum(r1_diff ** 2, dim=[1, 2])
                r2_penalty = torch.sum(r2_diff ** 2, dim=[1, 2])
                log_dict['r1_penalty'] = r1_penalty.mean().detach()
                log_dict['r2_penalty'] = r2_penalty.mean().detach()
                grad_penalty_loss = (r1_penalty.mean() + r2_penalty.mean()) / 2 * self.grad_penalty_weight
                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

            log_dict['disc_real_scores_mean'] = disc_scores_reals.mean().detach()
            log_dict['disc_fake_scores_mean'] = disc_scores_denoised.mean().detach()
            log_dict['disc_real_scores_std'] = disc_scores_reals.std().detach()
            log_dict['disc_fake_scores_std'] = disc_scores_denoised.std().detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised)

            if self.do_contrastive_disc:
                rolled_metadata = []
                for i in range(reals.shape[0]):
                    rolled_keys = self.roll_keys
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        if rolled_key in metadata[i]:
                            
                            rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)
                self._inject_disc_accomp(rolled_conditioning, shifted_accomp)
                if "accomp_input" in self.roll_keys:
                    # roll the accompaniment as well to match the rolled conditioning
                    rolled_accomp_input = rolled_conditioning['accomp_input'][0]
                    rolled_accomp_input = torch.roll(rolled_accomp_input, shifts=1, dims=0)
                    rolled_conditioning['accomp_input'] = [rolled_accomp_input]
                # with torch.compiler.disable():
                rolled_reals_gan_hidden_states = checkpoint(
                    self.discriminator, reals_t_gan, t_gan, cond=rolled_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer
                ).transpose(1, 2)
                rolled_reals_gan_hidden_states = rolled_reals_gan_hidden_states[:, :, self.context_size:]
                disc_scores_rolled_reals = checkpoint(self.discriminator_head, rolled_reals_gan_hidden_states)
                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals) * self.contrastive_loss_weight
                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss

            log_dict['train/dis_loss'] = loss_dis.detach()
            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()

            if is_accum_start:
                opt_disc.zero_grad()
            self.manual_backward(loss / grad_accum)

            if is_accum_end:
                if self.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()),
                        self.clip_grad_norm
                    )
                opt_disc.step()
                self._optim_step += 1

                if sched_disc is not None:
                    sched_disc.step()

        self._micro_step += 1

        log_dict['train/std_data'] = diffusion_input.std()
        log_dict['train/optim_step'] = self._optim_step

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss.detach()
