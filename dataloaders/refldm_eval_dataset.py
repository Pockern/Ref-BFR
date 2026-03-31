"""This file provides a deterministic Ref-LDM evaluation dataset for validation inside OSEDiff_refldm."""

import math
import os
from argparse import Namespace
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data as data

from dataloaders.refldm_dataset import normalize_image, read_image


class RefLDMEvalDataset(data.Dataset):
    """Build validation samples with fixed GT/LQ/reference paths and image names."""

    def __init__(self, dataset_config):
        """Load a validation CSV and resolve all file paths eagerly.

        Args:
            dataset_config: Namespace-like object with file_list, gt_dir, lq_dir, ref_dir,
                image_size, use_given_lq, max_num_refs, dup_to_max_num_refs and cat_refs_axis.
        """
        super().__init__()
        self.config = dataset_config
        self.image_size = tuple(dataset_config.image_size)
        self.use_given_lq = dataset_config.use_given_lq
        self.max_num_refs = dataset_config.max_num_refs
        self.dup_to_max_num_refs = dataset_config.dup_to_max_num_refs
        self.cat_refs_axis = dataset_config.cat_refs_axis

        dataframe = pd.read_csv(dataset_config.file_list)
        dataframe["image_name"] = dataframe["gt_image"].apply(lambda value: Path(value).name)
        dataframe["gt_image"] = dataframe["gt_image"].apply(
            lambda value: os.path.join(dataset_config.gt_dir, value)
        )
        if "lq_image" in dataframe.columns:
            dataframe["lq_image"] = dataframe["lq_image"].apply(
                lambda value: os.path.join(dataset_config.lq_dir, value)
            )
        else:
            dataframe["lq_image"] = dataframe["gt_image"]
        dataframe["ref_image"] = dataframe["ref_image"].apply(literal_eval)
        dataframe["ref_image"] = dataframe["ref_image"].apply(
            lambda values: [os.path.join(dataset_config.ref_dir, value) for value in values]
        )
        self.dataframe = dataframe

    def __len__(self):
        """Return the number of validation examples."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Return a deterministic validation sample with tensors still in numpy format."""
        row = self.dataframe.iloc[index]
        sample = {
            "image_name": row["image_name"],
            "gt_image": normalize_image(
                np.array(read_image(row["gt_image"], self.image_size), dtype=np.uint8)
            ).astype(np.float32),
        }

        if self.use_given_lq:
            lq_image = np.array(read_image(row["lq_image"], self.image_size), dtype=np.uint8)
        else:
            lq_image = np.array(read_image(row["gt_image"], self.image_size), dtype=np.uint8)
        sample["lq_image"] = normalize_image(lq_image).astype(np.float32)

        ref_paths = list(row["ref_image"])
        if self.max_num_refs is not None:
            if self.dup_to_max_num_refs:
                ref_paths = ref_paths * math.ceil(self.max_num_refs / len(ref_paths))
            ref_paths = ref_paths[: self.max_num_refs]

        ref_images = [
            normalize_image(np.array(read_image(ref_path, self.image_size), dtype=np.uint8)).astype(
                np.float32
            )
            for ref_path in ref_paths
        ]
        if self.cat_refs_axis == "width":
            sample["ref_image"] = np.concatenate(ref_images, axis=1)
        else:
            sample["ref_image"] = np.concatenate(ref_images, axis=2)
        return sample


def build_eval_dataset_config(global_args, dataset_config):
    """Merge global defaults into one dataset validation config."""
    return Namespace(
        file_list=dataset_config.file_list,
        gt_dir=dataset_config.gt_dir,
        lq_dir=dataset_config.lq_dir,
        ref_dir=dataset_config.ref_dir,
        image_size=getattr(dataset_config, "image_size", global_args.image_size),
        use_given_lq=getattr(dataset_config, "use_given_lq", True),
        max_num_refs=getattr(dataset_config, "max_num_refs", global_args.max_num_refs),
        dup_to_max_num_refs=getattr(
            dataset_config, "dup_to_max_num_refs", global_args.dup_to_max_num_refs
        ),
        cat_refs_axis=getattr(dataset_config, "cat_refs_axis", global_args.cat_refs_axis),
    )
