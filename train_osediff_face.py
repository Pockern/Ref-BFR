"""This file keeps the OSEDiff face-training entrypoint while switching to YAML-configured Ref-LDM distillation."""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from dataloaders.refldm_dataset import RefLDMFaceDataset
from osediff import OSEDiff_gen, OSEDiff_reg
from refldm.ldm.modules.losses.identity_loss import IdentityLoss
from utils.config import load_config, namespace_to_dict
from validation.runner import ValidationRunner


def flatten_tracker_config(config, prefix=""):
    """Flatten nested config values into TensorBoard-safe scalar entries.

    Args:
        config: Plain dictionary produced from the YAML namespace.
        prefix: Key prefix used during recursive flattening.

    Returns:
        A flat dictionary containing only scalar values accepted by tracker backends.
    """
    flat_config = {}
    for key, value in config.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat_config.update(flatten_tracker_config(value, prefix=flat_key))
        elif isinstance(value, list):
            # Serialize lists so tracker init keeps the information without rejecting the value.
            flat_config[flat_key] = str(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat_config[flat_key] = "None" if value is None else value
        else:
            flat_config[flat_key] = str(value)
    return flat_config


def save_training_state(
    checkpoint_path,
    model_gen,
    model_reg,
    optimizer,
    optimizer_reg,
    lr_scheduler,
    lr_scheduler_reg,
    epoch,
    global_step,
):
    """Save the minimal training state required to resume this script."""
    payload = {
        "generator_state_dict": model_gen.student_model.state_dict(),
        "reg_state_dict": model_reg.reg_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_reg_state_dict": optimizer_reg.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "lr_scheduler_reg_state_dict": lr_scheduler_reg.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(payload, checkpoint_path)


def load_training_state(
    checkpoint_path,
    model_gen,
    model_reg,
    optimizer,
    optimizer_reg,
    lr_scheduler,
    lr_scheduler_reg,
    device,
):
    """Load the minimal training state required to resume this script."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_gen.student_model.load_state_dict(checkpoint["generator_state_dict"], strict=False)
    model_reg.reg_model.load_state_dict(checkpoint["reg_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer_reg.load_state_dict(checkpoint["optimizer_reg_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    lr_scheduler_reg.load_state_dict(checkpoint["lr_scheduler_reg_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"]


def parse_args():
    """Parse the single YAML config entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_refldm_face.yaml",
        help="Path to the YAML config that replaces the original CLI arguments.",
    )
    return parser.parse_args()


def setup_training_logger(args, is_main_process):
    """Build a dedicated rank-0 logger for human-readable training progress.

    Args:
        args: Namespace-like training configuration.
        is_main_process: Whether the current process is allowed to emit logs.

    Returns:
        A configured logger for the main process, otherwise ``None``.
    """
    if not is_main_process:
        return None

    experiment_name = getattr(args, "tracker_project_name", "") or Path(args.output_dir).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.logging_dir)
    if not log_dir.is_absolute():
        log_dir = Path(args.output_dir) / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"train_osediff_face.{experiment_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{experiment_name}_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_log_message(event, **fields):
    """Convert structured logging fields into one deterministic text line.

    Args:
        event: High-level event name such as ``train`` or ``validation_start``.
        **fields: Key-value pairs to append to the line.

    Returns:
        A human-readable ``key=value`` string.
    """
    tokens = [event]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.6g}"
        tokens.append(f"{key}={value}")
    return " ".join(tokens)


def get_learning_rate(optimizer):
    """Read the current learning rate from the first optimizer parameter group."""
    return optimizer.param_groups[0]["lr"]


def build_identity_loss(args, device):
    """Create the face identity loss from the explicit training config path.

    Args:
        args: Namespace-like training configuration.
        device: Target torch device used by the current training process.

    Returns:
        A frozen ``IdentityLoss`` module placed on ``device``.
    """
    model_path = Path(args.identity_model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Identity model checkpoint not found: {model_path}")

    loss_module = IdentityLoss(
        model_path=str(model_path),
        center_crop=True,
        resize_hw=(112, 112),
    ).to(device)
    loss_module.requires_grad_(False)
    loss_module.eval()
    return loss_module


def main(args):
    """Train the copied OSEDiff framework with Ref-LDM teacher/student models."""
    tracker_root = getattr(args, "tracker_root", args.output_dir)
    logging_dir = Path(tracker_root, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=tracker_root, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    logger = setup_training_logger(args, accelerator.is_main_process)

    model_gen = OSEDiff_gen(args, device=accelerator.device)
    model_gen.set_train()
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()
    validation_runner = ValidationRunner(args, logger=logger)

    import lpips

    net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
    net_lpips.requires_grad_(False)
    net_identity = build_identity_loss(args, accelerator.device)

    if args.enable_xformers_memory_efficient_attention and not is_xformers_available():
        raise ValueError("xformers is not available, please install it before enabling the flag.")

    # missing
    # if args.gradient_checkpointing:
    #     model_gen.unet.enable_gradient_checkpointing()
    #     model_reg.unet_fix.enable_gradient_checkpointing()
    #     model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        list(model_gen.trainable_parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_reg = torch.optim.AdamW(
        list(model_reg.trainable_parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler_reg = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_reg,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = RefLDMFaceDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)
    net_identity = accelerator.prepare(net_identity)

    resume_path = getattr(args, "resume_path", "")
    start_epoch = 0
    global_step = 0
    if resume_path:
        start_epoch, global_step = load_training_state(
            checkpoint_path=resume_path,
            model_gen=accelerator.unwrap_model(model_gen),
            model_reg=accelerator.unwrap_model(model_reg),
            optimizer=optimizer,
            optimizer_reg=optimizer_reg,
            lr_scheduler=lr_scheduler,
            lr_scheduler_reg=lr_scheduler_reg,
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            logger.info(
                format_log_message(
                    "resume",
                    path=resume_path,
                    epoch=start_epoch,
                    global_step=global_step,
                )
            )

    if accelerator.is_main_process:
        tracker_config = flatten_tracker_config(namespace_to_dict(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        logger.info(
            format_log_message(
                "train_start",
                max_train_steps=args.max_train_steps,
                num_training_epochs=args.num_training_epochs,
                log_steps=getattr(args, "log_steps", 50),
                output_dir=args.output_dir,
            )
        )

    for epoch in range(start_epoch, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(model_gen, model_reg):
                gen_outputs = model_gen(batch=batch)
                if torch.cuda.device_count() > 1:
                    teacher_outputs = model_reg.module.teacher_supervision(
                        batch=batch,
                        noise=gen_outputs["noise"],
                        timestep=gen_outputs["timestep"],
                    )
                    loss_kl = (
                        model_reg.module.distribution_matching_loss(
                            gen_outputs["student_latent"], teacher_outputs["teacher_latent"]
                        )
                        * args.lambda_vsd
                    )
                    loss_d = model_reg.module.diff_loss(batch) * args.lambda_vsd_lora
                else:
                    teacher_outputs = model_reg.teacher_supervision(
                        batch=batch,
                        noise=gen_outputs["noise"],
                        timestep=gen_outputs["timestep"],
                    )
                    loss_kl = (
                        model_reg.distribution_matching_loss(
                            gen_outputs["student_latent"], teacher_outputs["teacher_latent"]
                        )
                        * args.lambda_vsd
                    )
                    loss_d = model_reg.diff_loss(batch) * args.lambda_vsd_lora

                loss_l2 = (
                    F.mse_loss(
                        gen_outputs["student_image"].float(), gen_outputs["target_image"].float()
                    )
                    * args.lambda_l2
                )
                loss_lpips = (
                    net_lpips(
                        gen_outputs["student_image"].float(), gen_outputs["target_image"].float()
                    ).mean()
                    * args.lambda_lpips
                )
                loss_id = (
                    net_identity(
                        gen_outputs["student_image"].float(), gen_outputs["target_image"].float()
                    ).mean()
                    * args.lambda_id
                )
                loss = loss_l2 + loss_lpips + loss_id + loss_kl

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_gen.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "train/lr": get_learning_rate(optimizer),
                        "train/loss": loss.detach().item(),
                        "train/loss_d": loss_d.detach().item(),
                        "train/loss_kl": loss_kl.detach().item(),
                        "train/loss_id": loss_id.detach().item(),
                        "train/loss_l2": loss_l2.detach().item(),
                        "train/loss_lpips": loss_lpips.detach().item(),
                    }

                    log_steps = max(int(getattr(args, "log_steps", 50)), 1)
                    if global_step % log_steps == 0:
                        logger.info(
                            format_log_message(
                                "train",
                                step=f"{global_step}/{args.max_train_steps}",
                                epoch=epoch,
                                lr=get_learning_rate(optimizer),
                                loss=logs["train/loss"],
                                loss_l2=logs["train/loss_l2"],
                                loss_lpips=logs["train/loss_lpips"],
                                loss_id=logs["train/loss_id"],
                                loss_kl=logs["train/loss_kl"],
                                loss_d=logs["train/loss_d"],
                            )
                        )

                    if global_step % args.checkpointing_steps == 1:
                        generator_outf = os.path.join(
                            args.output_dir, "checkpoints", f"model_{global_step}.pkl"
                        )
                        train_state_outf = os.path.join(
                            args.output_dir, "checkpoints", f"train_state_{global_step}.pt"
                        )
                        unwrapped_model_gen = accelerator.unwrap_model(model_gen)
                        unwrapped_model_reg = accelerator.unwrap_model(model_reg)
                        unwrapped_model_gen.save_model(generator_outf)
                        save_training_state(
                            checkpoint_path=train_state_outf,
                            model_gen=unwrapped_model_gen,
                            model_reg=unwrapped_model_reg,
                            optimizer=optimizer,
                            optimizer_reg=optimizer_reg,
                            lr_scheduler=lr_scheduler,
                            lr_scheduler_reg=lr_scheduler_reg,
                            epoch=epoch,
                            global_step=global_step,
                        )
                        logger.info(
                            format_log_message(
                                "checkpoint",
                                step=global_step,
                                generator_path=generator_outf,
                                state_path=train_state_outf,
                            )
                        )
                else:
                    logs = None

                if validation_runner.should_run(global_step):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        validation_logs = validation_runner.run(
                            model_gen=accelerator.unwrap_model(model_gen),
                            global_step=global_step,
                            device=accelerator.device,
                        )
                        if validation_logs:
                            logs.update(validation_logs)
                    accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        logger.info(
            format_log_message(
                "train_end",
                global_step=global_step,
                max_train_steps=args.max_train_steps,
            )
        )


if __name__ == "__main__":
    cli_args = parse_args()
    config_args = load_config(cli_args.config)
    main(config_args)
