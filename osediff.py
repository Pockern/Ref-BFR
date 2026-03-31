"""This file keeps the OSEDiff module layout while swapping in Ref-LDM teacher/student models."""

import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REFLDM_DIR = os.path.join(PROJECT_DIR, "refldm")
if REFLDM_DIR not in sys.path:
    sys.path.append(REFLDM_DIR)

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def _build_refldm_model(config_path, teacher_ckpt_path, vae_ckpt_path, device, trainable):
    """Instantiate a Ref-LDM model and load the requested checkpoints.

    Args:
        config_path: Path to the Ref-LDM YAML config.
        teacher_ckpt_path: Path to the source Ref-LDM checkpoint.
        vae_ckpt_path: Path to the VQ/VAE checkpoint.
        device: Target device for the created model.
        trainable: Whether the diffusion backbone should be trainable.

    Returns:
        A configured Ref-LDM LatentDiffusion model on the requested device.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    model_config["model"]["params"]["first_stage_config"]["params"]["ckpt_path"] = vae_ckpt_path
    for key in ["ckpt_path", "perceptual_loss_config"]:
        model_config["model"]["params"].pop(key, None)

    model = instantiate_from_config(model_config["model"])
    state_dict = torch.load(teacher_ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # The migration baseline keeps the latent space fixed and only trains
    # the diffusion backbone that predicts the one-step latent update.
    model.requires_grad_(False)
    if trainable:
        model.model.diffusion_model.requires_grad_(True)
        model.model.diffusion_model.train()
    else:
        model.eval()
    return model


def _batch_to_condition(batch, device):
    """Move raw conditioning tensors onto the target device."""
    return {
        "lq_image": batch["lq_image"].to(device),
        "ref_image": batch["ref_image"].to(device),
    }


def _clone_uncond_condition(cond):
    """Create the unconditional branch used by Ref-LDM CFG."""
    unconditional = {key: value.detach().clone() for key, value in cond.items()}
    if "ref_image" in unconditional:
        unconditional["ref_image"].zero_()
    return unconditional


class RefLDMOneStepMixin:
    """Utility methods shared across the OSEDiff-style wrappers."""

    @staticmethod
    def _prepare_inputs(model, batch):
        """Encode GT and conditions through the original Ref-LDM data path."""
        z, raw_cond, image, _ = model.get_input(
            batch, model.first_stage_key, return_first_stage_outputs=True
        )
        encoded_cond = model.get_learned_conditioning(raw_cond)
        return z, raw_cond, encoded_cond, image

    @staticmethod
    def _one_step_predict(model, x_t, cond, timestep):
        """Run the Ref-LDM backbone in one-step mode and recover x0."""
        eps_pred = model.apply_model(x_t, timestep, cond)
        x0_pred = model.predict_start_from_noise(x_t, timestep, eps_pred)
        image_pred = model.decode_first_stage(x0_pred)
        return image_pred, x0_pred, eps_pred

    @staticmethod
    def _latent_shape_from_model(model):
        """Return the latent shape used by Ref-LDM DDIM sampling."""
        channels = model.model.diffusion_model.out_channels
        return [channels, model.image_size, model.image_size]


class OSEDiff_gen(nn.Module, RefLDMOneStepMixin):
    """Generator wrapper that preserves the OSEDiff role with a Ref-LDM student."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student_model = _build_refldm_model(
            config_path=args.teacher_config_path,
            teacher_ckpt_path=args.student_init_ckpt_path or args.teacher_ckpt_path,
            vae_ckpt_path=args.vae_ckpt_path,
            device=self.device,
            trainable=True,
        )

    def set_train(self):
        """Enable gradients only for the Ref-LDM diffusion backbone."""
        self.student_model.train()
        self.student_model.first_stage_model.eval()
        if hasattr(self.student_model, "cond_stage_model"):
            cond_stage_model = self.student_model.cond_stage_model
            if isinstance(cond_stage_model, dict):
                for module in cond_stage_model.values():
                    module.eval()

    def trainable_parameters(self):
        """Return the parameters optimized by the generator optimizer."""
        return self.student_model.model.diffusion_model.parameters()

    def forward(self, batch, noise=None):
        """Run one-step student prediction on a Ref-LDM training batch."""
        z, raw_cond, encoded_cond, target_image = self._prepare_inputs(self.student_model, batch)
        batch_size = z.shape[0]
        timestep = torch.full(
            (batch_size,), self.args.student_timestep, device=z.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(z)
        x_t = self.student_model.q_sample(x_start=z, t=timestep, noise=noise)
        student_image, student_x0, student_eps = self._one_step_predict(
            self.student_model, x_t, encoded_cond, timestep
        )
        return {
            "student_image": student_image,
            "student_latent": student_x0,
            "student_eps": student_eps,
            "target_latent": z,
            "target_image": target_image,
            "raw_cond": raw_cond,
            "encoded_cond": encoded_cond,
            "timestep": timestep,
            "noise": noise,
            "x_t": x_t,
        }

    def save_model(self, outf):
        """Persist the student checkpoint in a single self-contained file."""
        payload = {
            "state_dict": self.student_model.state_dict(),
            "teacher_config_path": self.args.teacher_config_path,
            "teacher_ckpt_path": self.args.teacher_ckpt_path,
            "vae_ckpt_path": self.args.vae_ckpt_path,
            "student_timestep": self.args.student_timestep,
        }
        torch.save(payload, outf)


class OSEDiff_reg(nn.Module, RefLDMOneStepMixin):
    """Regularization wrapper that keeps the OSEDiff gen/reg split intact."""

    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.device = accelerator.device
        self.teacher_model = _build_refldm_model(
            config_path=args.teacher_config_path,
            teacher_ckpt_path=args.teacher_ckpt_path,
            vae_ckpt_path=args.vae_ckpt_path,
            device=self.device,
            trainable=False,
        )
        self.teacher_sampler = DDIMSampler(
            self.teacher_model, print_tqdm=getattr(args, "print_teacher_progress", False)
        )
        self.reg_model = _build_refldm_model(
            config_path=args.teacher_config_path,
            teacher_ckpt_path=args.student_init_ckpt_path or args.teacher_ckpt_path,
            vae_ckpt_path=args.vae_ckpt_path,
            device=self.device,
            trainable=True,
        )

    def set_train(self):
        """Place the auxiliary Ref-LDM branch in training mode."""
        self.reg_model.train()
        self.reg_model.first_stage_model.eval()

    def trainable_parameters(self):
        """Return the parameters optimized by the reg optimizer."""
        return self.reg_model.model.diffusion_model.parameters()

    @torch.no_grad()
    def teacher_supervision(self, batch, noise, timestep):
        """Run the frozen multi-step teacher using the same noisy latent seed."""
        z, raw_cond, encoded_cond, _ = self._prepare_inputs(self.teacher_model, batch)
        uc = None
        if self.args.cfg_scale != 1.0:
            uc = _clone_uncond_condition(encoded_cond)

        # Recreate the exact noisy latent used by the student before teacher DDIM.
        x_t = self.teacher_model.q_sample(x_start=z, t=timestep, noise=noise)
        teacher_context = self.teacher_model.ema_scope() if self.teacher_model.use_ema else nullcontext()
        with teacher_context:
            teacher_latent, _ = self.teacher_sampler.sample(
                S=self.args.teacher_ddim_steps,
                conditioning=encoded_cond,
                unconditional_guidance_scale=self.args.cfg_scale,
                unconditional_conditioning=uc,
                shape=list(z.shape[1:]),
                x_T=x_t,
                batch_size=z.shape[0],
                verbose=False,
            )
        teacher_image = self.teacher_model.decode_first_stage(teacher_latent)
        return {
            "teacher_latent": teacher_latent.detach(),
            "teacher_image": teacher_image.detach(),
        }

    def distribution_matching_loss(self, student_latents, teacher_latents):
        """Match the one-step student latent to the frozen teacher output."""
        return F.mse_loss(student_latents.float(), teacher_latents.float(), reduction="mean")

    def diff_loss(self, batch):
        """Train the auxiliary branch with the original Ref-LDM diffusion loss."""
        z, raw_cond, encoded_cond, image = self._prepare_inputs(self.reg_model, batch)
        timestep = torch.randint(
            0, self.reg_model.num_timesteps, (z.shape[0],), device=z.device
        ).long()
        loss_d, _ = self.reg_model.p_losses(z, encoded_cond, timestep, image)
        return loss_d


class OSEDiff_test(nn.Module, RefLDMOneStepMixin):
    """Inference wrapper that keeps the OSEDiff testing entrypoint."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.osediff_path, map_location="cpu")
        config_path = checkpoint.get("teacher_config_path", args.teacher_config_path)
        teacher_ckpt_path = checkpoint.get("teacher_ckpt_path", args.teacher_ckpt_path)
        vae_ckpt_path = checkpoint.get("vae_ckpt_path", args.vae_ckpt_path)
        self.student_timestep = checkpoint.get("student_timestep", args.student_timestep)
        self.student_model = _build_refldm_model(
            config_path=config_path,
            teacher_ckpt_path=teacher_ckpt_path,
            vae_ckpt_path=vae_ckpt_path,
            device=self.device,
            trainable=False,
        )
        self.student_model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.student_model.eval()

    @torch.no_grad()
    def forward(self, lq, ref):
        """Run one-step Ref-LDM inference with explicit LQ and reference tensors."""
        cond = self.student_model.get_learned_conditioning(
            {
                "lq_image": lq.to(self.device),
                "ref_image": ref.to(self.device),
            }
        )
        latent_shape = self._latent_shape_from_model(self.student_model)
        timestep = torch.full((lq.shape[0],), self.student_timestep, device=self.device, dtype=torch.long)
        x_t = torch.randn((lq.shape[0], *latent_shape), device=self.device)
        image_pred, _, _ = self._one_step_predict(self.student_model, x_t, cond, timestep)
        return image_pred


class OSEDiff_inference_time(OSEDiff_test):
    """Reuse the test wrapper for timing benchmarks."""
