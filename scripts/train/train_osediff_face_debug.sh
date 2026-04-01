#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2,3"

cd /mnt/data/shenglong/project/OSEDiff_refldm
accelerate launch train_osediff_face.py --config configs/train/train_refldm_face_debug.yaml
