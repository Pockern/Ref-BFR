"""This file implements optional validation metrics that mirror the Ref-LDM evaluation script."""

from collections import OrderedDict

import torch


class ValidationMetricCollection:
    """Lazily build only the metrics enabled by the validation YAML config."""

    def __init__(self, metrics_config, device):
        """Create metric instances on demand.

        Args:
            metrics_config: Nested namespace with one block per metric and an enabled flag.
            device: Torch device used to evaluate all metrics.
        """
        self.metrics_config = metrics_config
        self.device = device
        self.metrics = OrderedDict()
        self.fid_metric = None

        if self.is_enabled("ids"):
            from refldm.ldm.modules.losses.identity_loss import IdentityLoss

            self.metrics["ids"] = IdentityLoss(
                model_path=metrics_config.ids.model_path,
            ).to(device)

        enabled_pyiqa_metrics = {
            "lpips": "lpips-vgg",
            "psnr": "psnr",
            "ssim": "ssim",
            "niqe": "niqe",
            "musiq": "musiq",
        }
        if any(self.is_enabled(name) for name in enabled_pyiqa_metrics):
            import pyiqa

            for metric_name, pyiqa_name in enabled_pyiqa_metrics.items():
                if self.is_enabled(metric_name):
                    self.metrics[metric_name] = pyiqa.create_metric(
                        pyiqa_name, device=device
                    )
            if self.is_enabled("fid"):
                self.fid_metric = pyiqa.create_metric("fid", device=device)

    def is_enabled(self, metric_name):
        """Return whether one metric is enabled in config."""
        metric_config = getattr(self.metrics_config, metric_name, None)
        return bool(metric_config and getattr(metric_config, "enabled", False))

    @staticmethod
    def to_zero_one(image):
        """Convert [-1, 1] tensors into [0, 1] tensors for pyiqa metrics."""
        return ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def evaluate_batch(self, pred_image, gt_image):
        """Evaluate all non-FID metrics for one batch and return summed values."""
        results = {}
        pred_zero_one = self.to_zero_one(pred_image)
        gt_zero_one = self.to_zero_one(gt_image)
        batch_size = pred_image.shape[0]

        with torch.no_grad():
            for metric_name, metric in self.metrics.items():
                if metric_name == "ids":
                    value = (1.0 - metric(pred_image, gt_image)).mean()
                elif metric_name in {"niqe", "musiq"}:
                    value = metric(pred_zero_one).mean()
                else:
                    value = metric(pred_zero_one, gt_zero_one).mean()
                results[metric_name] = float(value.item()) * batch_size
        return results

    def evaluate_fid(self, gt_dir, pred_dir, fid_kwargs=None):
        """Run the optional FID metric after all images have been saved."""
        if self.fid_metric is None:
            return None
        fid_kwargs = fid_kwargs or {}
        with torch.no_grad():
            if fid_kwargs:
                return float(self.fid_metric(pred_dir, **fid_kwargs).item())
            return float(self.fid_metric(gt_dir, pred_dir).item())
