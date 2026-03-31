"""This file provides the Ref-LDM restoration dataset in the OSEDiff dataloaders layout."""

import math
import os
import random
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import binary_dilation

from refldm.ldm.data import degradations


def normalize_image(image):
    """Convert an RGB image from [0, 255] to [-1, 1]."""
    return image / 127.5 - 1.0


def read_image(image_path, resize=None):
    """Read an RGB image and optionally resize it to the expected training size."""
    image = Image.open(str(image_path)).convert("RGB")
    if resize is not None and image.size != resize:
        image = image.resize(resize)
    return image


def sample_degraded_image(
    gt_image,
    blur_kernel_list,
    blur_kernel_prob,
    blur_kernel_size,
    blur_sigma,
    downsample_range,
    noise_range,
    jpeg_range,
):
    """Apply the Ref-LDM synthetic degradation pipeline to build the LQ input."""
    lq_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR) / 255.0
    height, width, _ = gt_image.shape

    kernel = degradations.random_mixed_kernels(
        blur_kernel_list,
        blur_kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma,
    )
    lq_image = cv2.filter2D(lq_image, -1, kernel)
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    lq_image = cv2.resize(
        lq_image, (int(width // scale), int(height // scale)), interpolation=cv2.INTER_LINEAR
    )
    if noise_range is not None:
        lq_image = degradations.random_add_gaussian_noise(lq_image, noise_range)
    if jpeg_range is not None:
        lq_image = degradations.random_add_jpg_compression(lq_image, jpeg_range)
    lq_image = cv2.resize(lq_image, (width, height), interpolation=cv2.INTER_LINEAR)
    lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)
    return np.clip((lq_image * 255).round(), 0, 255)


class RefLDMFaceDataset(data.Dataset):
    """Serve Ref-LDM restoration batches in the format expected by the copied OSEDiff loop."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.split = split
        self.image_size = tuple(args.image_size)
        self.use_given_lq = args.use_given_lq
        self.use_given_ref = True
        self.max_num_refs = args.max_num_refs
        self.dup_to_max_num_refs = args.dup_to_max_num_refs
        self.shuffle_refs_prob = args.shuffle_refs_prob
        self.cat_refs_axis = args.cat_refs_axis
        raw_loss_weight = getattr(args, "loss_weight_by_semantic", None)
        self.loss_weight_by_semantic = (
            {int(key): value for key, value in vars(raw_loss_weight).items()}
            if hasattr(raw_loss_weight, "__dict__")
            else raw_loss_weight
        )
        self.semantic_dir = getattr(args, "semantic_dir", "")
        self.lr_flip_aug = args.lr_flip_aug if split == "train" else False
        self.degrad_opt = vars(args.degrad_opt) if hasattr(args.degrad_opt, "__dict__") else args.degrad_opt

        dataframe = pd.read_csv(args.file_list)
        dataframe["gt_image"] = dataframe["gt_image"].apply(lambda value: os.path.join(args.gt_dir, value))
        if self.use_given_lq and "lq_image" in dataframe.columns:
            dataframe["lq_image"] = dataframe["lq_image"].apply(
                lambda value: os.path.join(args.lq_dir, value)
            )
        dataframe["ref_image"] = dataframe["ref_image"].apply(literal_eval)
        dataframe["ref_image"] = dataframe["ref_image"].apply(
            lambda values: [os.path.join(args.ref_dir, value) for value in values]
        )
        self.dataframe = dataframe

        if split == "train" and args.ref_rand_aug:
            self.ref_rand_aug = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                    ),
                    transforms.RandomAffine(
                        degrees=2,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomPerspective(
                        distortion_scale=0.2,
                        p=1.0,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.ref_rand_aug = None

    def __len__(self):
        """Return the number of restoration examples."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Build a single Ref-LDM sample with GT, LQ and concatenated references."""
        row = self.dataframe.iloc[index]
        sample = {}

        gt_image = np.array(read_image(row["gt_image"], self.image_size), dtype=np.uint8)
        sample["gt_image"] = normalize_image(gt_image).astype(np.float32)

        if self.loss_weight_by_semantic is not None:
            gt_name = Path(row["gt_image"]).stem
            semantic_map = np.load(os.path.join(self.semantic_dir, f"{gt_name}.npy"))
            sample["loss_weight_map"] = self.create_loss_weight_map_from_semantic_map(semantic_map)

        if self.use_given_lq:
            lq_image = np.array(read_image(row["lq_image"], self.image_size), dtype=np.uint8)
        else:
            lq_image = sample_degraded_image(gt_image, **self.degrad_opt)
        sample["lq_image"] = normalize_image(lq_image).astype(np.float32)

        ref_paths = list(row["ref_image"])
        if self.shuffle_refs_prob > random.random():
            random.shuffle(ref_paths)
        if self.max_num_refs is not None:
            if self.dup_to_max_num_refs:
                ref_paths = ref_paths * math.ceil(self.max_num_refs / len(ref_paths))
            ref_paths = ref_paths[: self.max_num_refs]

        ref_images = []
        for ref_path in ref_paths:
            ref_image = read_image(ref_path, self.image_size)
            if self.ref_rand_aug is not None:
                ref_image = self.ref_rand_aug(ref_image)
            ref_images.append(normalize_image(np.array(ref_image, dtype=np.uint8)).astype(np.float32))

        if self.cat_refs_axis == "width":
            sample["ref_image"] = np.concatenate(ref_images, axis=1)
        else:
            sample["ref_image"] = np.concatenate(ref_images, axis=2)

        if self.lr_flip_aug and random.random() > 0.5:
            for key, value in sample.items():
                sample[key] = np.fliplr(value).copy()

        return sample

    def create_loss_weight_map_from_semantic_map(self, semantic_map, dilation=5):
        """Expand semantic masks into a weighted map used by Ref-LDM losses."""
        weight_map = np.zeros_like(semantic_map)
        for weight, class_ids in sorted(self.loss_weight_by_semantic.items()):
            for class_id in class_ids:
                binary_map = semantic_map == class_id
                binary_map = binary_dilation(binary_map, iterations=dilation)
                weight_map[binary_map] = weight
        return weight_map.astype(np.float32)
