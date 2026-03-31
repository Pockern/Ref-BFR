"""This file adds a standalone validation runner that can be called from the training loop."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataloaders.refldm_eval_dataset import RefLDMEvalDataset, build_eval_dataset_config
from validation.metrics import ValidationMetricCollection


class ValidationRunner:
    """Run periodic validation without changing the existing training or loss logic."""

    def __init__(self, args):
        """Build all configured validation datasets and metrics once.

        Args:
            args: Top-level training config namespace.
        """
        self.args = args
        self.validation_config = getattr(args, "validation", None)
        self.enabled = bool(self.validation_config and getattr(self.validation_config, "enabled", False))
        self.datasets = {}
        self.output_dir = Path(args.output_dir) / "eval"
        self.summary_path = self.output_dir / getattr(
            self.validation_config, "summary_filename", "validation_summary.csv"
        ) if self.enabled else None
        self.metrics = None

        if not self.enabled:
            return

        dataset_items = vars(self.validation_config.datasets).items()
        for dataset_name, dataset_config in dataset_items:
            if not getattr(dataset_config, "enabled", False):
                continue
            merged_config = build_eval_dataset_config(args, dataset_config)
            self.datasets[dataset_name] = {
                "dataset": RefLDMEvalDataset(merged_config),
                "config": dataset_config,
            }

    def should_run(self, global_step):
        """Return whether validation should run at the current training step."""
        if not self.enabled or not self.datasets:
            return False
        every_n_steps = getattr(self.validation_config, "every_n_steps", 0)
        return bool(every_n_steps and global_step % every_n_steps == 0)

    def _ensure_metrics(self, device):
        """Create metric objects lazily on the validation device."""
        if self.metrics is None:
            self.metrics = ValidationMetricCollection(self.validation_config.metrics, device)

    @staticmethod
    def _latent_shape_from_student(student_model):
        """Recover the latent tensor shape expected by the one-step student."""
        return [
            student_model.model.diffusion_model.out_channels,
            student_model.image_size,
            student_model.image_size,
        ]

    def _predict_batch(self, student_model, batch, generator, device, student_timestep):
        """Run one-step inference with the in-memory student model."""
        lq_image = batch["lq_image"].to(device)
        ref_image = batch["ref_image"].to(device)
        cond = student_model.get_learned_conditioning(
            {
                "lq_image": lq_image,
                "ref_image": ref_image,
            }
        )
        latent_shape = self._latent_shape_from_student(student_model)
        timestep = torch.full(
            (lq_image.shape[0],), student_timestep, device=device, dtype=torch.long
        )
        x_t = torch.randn(
            (lq_image.shape[0], *latent_shape), generator=generator, device=device
        )
        eps_pred = student_model.apply_model(x_t, timestep, cond)
        x0_pred = student_model.predict_start_from_noise(x_t, timestep, eps_pred)
        return student_model.decode_first_stage(x0_pred)

    def _save_batch_images(self, pred_image, image_names, pred_dir):
        """Persist predicted images so validation can mirror the original eval workflow."""
        pred_zero_one = ((pred_image + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        for image_tensor, image_name in zip(pred_zero_one, image_names):
            to_pil_image(image_tensor).save(pred_dir / image_name)

    def _append_summary_rows(self, rows):
        """Append validation rows to a CSV summary file."""
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step",
            "dataset",
            "num_images",
            "ids",
            "lpips",
            "psnr",
            "ssim",
            "niqe",
            "musiq",
            "fid",
        ]
        write_header = not self.summary_path.exists()
        with open(self.summary_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def run(self, model_gen, global_step, device):
        """Run validation for all enabled datasets and return one flat metrics dict."""
        if not self.enabled or not self.datasets:
            return {}

        self._ensure_metrics(device)
        model_gen.eval()
        student_model = model_gen.student_model
        student_model.eval()

        step_dir = self.output_dir / f"step_{global_step:06d}"
        seed = getattr(self.validation_config, "seed", 123)
        batch_size = getattr(self.validation_config, "batch_size", 1)
        num_workers = getattr(self.validation_config, "num_workers", 0)
        max_batches = getattr(self.validation_config, "max_batches", None)
        save_images = getattr(self.validation_config, "save_images", True)

        summary_rows = []
        flat_metrics = {}
        for dataset_name, dataset_bundle in self.datasets.items():
            dataset = dataset_bundle["dataset"]
            dataset_config = dataset_bundle["config"]
            pred_dir = step_dir / dataset_name
            if save_images:
                pred_dir.mkdir(parents=True, exist_ok=True)

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            metric_sums = {metric_name: 0.0 for metric_name in self.metrics.metrics.keys()}
            image_count = 0

            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                pred_image = self._predict_batch(
                    student_model=student_model,
                    batch=batch,
                    generator=generator,
                    device=device,
                    student_timestep=getattr(model_gen.args, "student_timestep"),
                )
                gt_image = batch["gt_image"].to(device)
                batch_size_current = gt_image.shape[0]

                if save_images:
                    self._save_batch_images(pred_image, batch["image_name"], pred_dir)
                batch_metrics = self.metrics.evaluate_batch(pred_image, gt_image)
                for metric_name, metric_value in batch_metrics.items():
                    metric_sums[metric_name] += metric_value
                image_count += batch_size_current

            if image_count == 0:
                continue
            dataset_results = {
                metric_name: metric_sums[metric_name] / image_count
                for metric_name in metric_sums
            }
            fid_kwargs = vars(dataset_config.fid_kwargs) if hasattr(dataset_config, "fid_kwargs") else {}
            fid_value = None
            if save_images and self.metrics.is_enabled("fid"):
                fid_value = self.metrics.evaluate_fid(
                    gt_dir=dataset_config.gt_dir,
                    pred_dir=str(pred_dir),
                    fid_kwargs=fid_kwargs,
                )
                if fid_value is not None:
                    dataset_results["fid"] = fid_value

            summary_row = {
                "step": global_step,
                "dataset": dataset_name,
                "num_images": image_count,
                "ids": dataset_results.get("ids"),
                "lpips": dataset_results.get("lpips"),
                "psnr": dataset_results.get("psnr"),
                "ssim": dataset_results.get("ssim"),
                "niqe": dataset_results.get("niqe"),
                "musiq": dataset_results.get("musiq"),
                "fid": dataset_results.get("fid"),
            }
            summary_rows.append(summary_row)

            for metric_name, metric_value in dataset_results.items():
                flat_metrics[f"val/{dataset_name}/{metric_name}"] = metric_value

        self._append_summary_rows(summary_rows)
        if hasattr(model_gen, "set_train"):
            model_gen.set_train()
        else:
            model_gen.train()
        return flat_metrics
