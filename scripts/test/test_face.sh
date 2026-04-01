#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="7"

cd /mnt/data/shenglong/project/OSEDiff_refldm
# python test_osediff.py --config configs/test_refldm_face.yaml
# python test_osediff.py --config configs/test_ffhq_ref_moderate.yaml
# python test_osediff.py --config configs/test_ffhq_ref_severe.yaml
python test_osediff.py --config configs/test/test_celeba_test_ref.yaml
