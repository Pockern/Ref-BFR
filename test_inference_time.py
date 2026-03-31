"""This file benchmarks the copied OSEDiff_refldm one-step inference path from YAML config."""

import argparse
import time

import torch
from tqdm import tqdm

from osediff import OSEDiff_inference_time
from utils.config import load_config


def parse_args():
    """Parse the YAML config path for the timing benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_refldm_face.yaml",
        help="Path to the YAML config used for the timing benchmark.",
    )
    return parser.parse_args()


def main(args):
    """Run a simple timing benchmark over random one-step Ref-LDM inputs."""
    model = OSEDiff_inference_time(args)
    model.eval()

    latent_size = tuple(args.image_size)
    total_time = 0.0
    warmup_iterations = args.warmup_iterations
    inference_iterations = args.inference_iterations

    lq = torch.randn((1, 3, *latent_size), device=model.device)
    ref = torch.randn((1, 3, latent_size[0], latent_size[1] * args.max_num_refs), device=model.device)

    print(f"Running {warmup_iterations} warm-up iterations...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(lq, ref)

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Starting inference for {inference_iterations} iterations...")
    for _ in tqdm(range(inference_iterations), desc="Inference"):
        start_time = time.time()
        with torch.no_grad():
            _ = model(lq, ref)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start_time

    avg_time = total_time / inference_iterations
    print(f"Average inference time per iteration: {avg_time:.4f} seconds.")


if __name__ == "__main__":
    cli_args = parse_args()
    config_args = load_config(cli_args.config)
    main(config_args)

