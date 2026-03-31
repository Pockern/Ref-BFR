"""This file keeps the OSEDiff face-training entrypoint while switching to YAML-configured Ref-LDM distillation."""

import argparse
import os
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
from tqdm.auto import tqdm

from dataloaders.refldm_dataset import RefLDMFaceDataset
from osediff import OSEDiff_gen, OSEDiff_reg
from utils.config import load_config, namespace_to_dict
from validation.runner import ValidationRunner


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


def main(args):
    """Train the copied OSEDiff framework with Ref-LDM teacher/student models."""
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
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

    model_gen = OSEDiff_gen(args)
    model_gen.set_train()
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()
    validation_runner = ValidationRunner(args)

    import lpips

    net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
    net_lpips.requires_grad_(False)

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

    if accelerator.is_main_process:
        tracker_config = namespace_to_dict(args)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for epoch in range(args.num_training_epochs):
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
                        gen_outputs["student_image"].float(), teacher_outputs["teacher_image"].float()
                    )
                    * args.lambda_l2
                )
                loss_lpips = (
                    net_lpips(
                        gen_outputs["student_image"].float(), gen_outputs["target_image"].float()
                    ).mean()
                    * args.lambda_lpips
                )
                loss = loss_l2 + loss_lpips + loss_kl

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
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "train/loss_d": loss_d.detach().item(),
                        "train/loss_kl": loss_kl.detach().item(),
                        "train/loss_l2": loss_l2.detach().item(),
                        "train/loss_lpips": loss_lpips.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)
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
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    cli_args = parse_args()
    config_args = load_config(cli_args.config)
    main(config_args)
