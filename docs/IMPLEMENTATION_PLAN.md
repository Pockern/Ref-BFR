This file records the implemented migration plan for the copied OSEDiff_refldm project.

# Ref-LDM in Copied OSEDiff Framework

## Summary

- `OSEDiff/` was copied into `OSEDiff_refldm/` as the project base.
- YAML config files now replace the original long CLI argument lists.
- The OSEDiff wrapper roles are preserved:
  - `OSEDiff_gen` is the one-step student branch.
  - `OSEDiff_reg` is the auxiliary regularization branch.
  - `OSEDiff_test` remains the inference entrypoint.
- Ref-LDM provides the latent space, dual conditioning path, teacher checkpoint, and DDIM teacher sampling.

## Key Implementation Choices

- The project keeps the OSEDiff training entrypoint style and dual-optimizer loop.
- Only task-specific components were swapped:
  - data loading
  - backbone internals
  - teacher/student supervision flow
  - inference inputs
- The student and reg branches train only the Ref-LDM diffusion backbone so the original VQ latent space stays fixed.

## Config Extraction

- `utils/config.py` loads YAML files into an argparse-style namespace.
- `train_osediff_face.py`, `test_osediff.py`, and `test_inference_time.py` now only accept `--config`.
- Config keys intentionally mirror the original OSEDiff naming where possible.

## New Files

- `configs/train_refldm_face.yaml`
- `configs/test_refldm_face.yaml`
- `dataloaders/refldm_dataset.py`
- `utils/config.py`
- `refldm/`

## Verification Scope

- Config parsing and module imports were targeted for smoke checking.
- Full training and inference still depend on local availability of the copied Ref-LDM and OSEDiff runtime dependencies plus the configured checkpoints and dataset paths.

