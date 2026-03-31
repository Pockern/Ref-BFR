"""This file keeps the OSEDiff test entrypoint while running one-step Ref-LDM inference from YAML config."""

import argparse
import os

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


if __name__ == "__main__":
    cli_args = parse_args()
    args = load_config(cli_args.config)
    model = OSEDiff_test(args)

    os.makedirs(args.output_dir, exist_ok=True)
    lq = read_and_normalize_image(args.input_image, args.image_size).unsqueeze(0).to(model.device)
    ref = build_reference_tensor(args.reference_images, args.image_size).unsqueeze(0).to(model.device)

    with torch.no_grad():
        output_image = model(lq, ref)
    output_image = ((output_image + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(0).cpu()

    output_path = os.path.join(args.output_dir, args.output_name)
    to_pil_image(output_image).save(output_path)
    print(f"Saved output to {output_path}")

