"""This file keeps the OSEDiff test entrypoint while running generator-only Ref-LDM inference from YAML config."""

import argparse
import csv
import os
import time
from ast import literal_eval
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from osediff import OSEDiff_test
from utils.config import load_config


def parse_args():
    """Parse the YAML config path for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_refldm_face.yaml",
        help="Path to the YAML config used for one-step Ref-LDM inference.",
    )
    return parser.parse_args()


def read_and_normalize_image(image_path, resize):
    """Read an RGB image, resize it, and normalize it to [-1, 1]."""
    image = Image.open(str(image_path)).convert("RGB")
    if resize is not None and image.size != tuple(resize):
        image = image.resize(tuple(resize))
    return pil_to_tensor(image) / 127.5 - 1.0


def build_reference_tensor(reference_paths, resize):
    """Concatenate all references along width to match the Ref-LDM conditioning contract."""
    ref_images = [read_and_normalize_image(path, resize) for path in reference_paths]
    return torch.cat(ref_images, dim=-1)


def build_single_job(args):
    """Create one inference job from the single-image config fields."""
    return [
        {
            "image_name": args.output_name,
            "lq_path": args.input_image,
            "ref_paths": list(args.reference_images),
        }
    ]


def build_dataset_jobs(args):
    """Create inference jobs from a Ref-LDM-style CSV mapping file."""
    jobs = []
    with open(args.test_file_list, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ref_names = literal_eval(row["ref_image"])
            if args.max_num_refs is not None:
                ref_names = ref_names[: args.max_num_refs]
            jobs.append(
                {
                    "image_name": Path(row["gt_image"]).name,
                    "lq_path": os.path.join(args.test_lq_dir, row["lq_image"]),
                    "ref_paths": [os.path.join(args.test_ref_dir, name) for name in ref_names],
                }
            )
    return jobs


def build_inference_jobs(args):
    """Return either one single-image job or one list of dataset jobs."""
    input_mode = getattr(args, "input_mode", "single")
    if input_mode == "single":
        return build_single_job(args)
    if input_mode == "dataset_csv":
        return build_dataset_jobs(args)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def get_inference_step_count(args):
    """Return the configured diffusion step count for reporting."""
    if getattr(args, "inference_mode", "one_step") == "ddim_multi_step":
        return int(getattr(args, "ddim_step", 0))
    return 1


if __name__ == "__main__":
    cli_args = parse_args()
    args = load_config(cli_args.config)
    model = OSEDiff_test(args)

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = build_inference_jobs(args)
    print(f"Running inference for {len(jobs)} image(s).")
    total_elapsed_time = 0.0
    total_step_count = 0
    configured_step_count = get_inference_step_count(args)

    for job_index, job in enumerate(jobs, start=1):
        print(f"[{job_index}/{len(jobs)}] Running inference for {job['image_name']}")
        lq = read_and_normalize_image(job["lq_path"], args.image_size).unsqueeze(0).to(model.device)
        ref = build_reference_tensor(job["ref_paths"], args.image_size).unsqueeze(0).to(model.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output_image = model(lq, ref)
        elapsed_time = time.perf_counter() - start_time
        total_elapsed_time += elapsed_time
        total_step_count += configured_step_count
        output_image = ((output_image + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(0).cpu()

        output_path = os.path.join(args.output_dir, job["image_name"])
        to_pil_image(output_image).save(output_path)
        print(
            f"[{job_index}/{len(jobs)}] Saved output to {output_path} "
            f"(time={elapsed_time:.3f}s, steps={configured_step_count})"
        )

    if jobs:
        avg_elapsed_time = total_elapsed_time / len(jobs)
        avg_step_count = total_step_count / len(jobs)
        print(
            "Inference summary: "
            f"images={len(jobs)}, avg_time_per_image={avg_elapsed_time:.3f}s, "
            f"avg_steps_per_image={avg_step_count:.2f}"
        )
