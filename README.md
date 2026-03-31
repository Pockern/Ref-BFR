This file documents the copied OSEDiff_refldm project and its Ref-LDM-specific entrypoints.

# OSEDiff_refldm

`OSEDiff_refldm/` is a direct copy of `OSEDiff/` with minimal task-specific changes for Ref-LDM one-step distillation.

## What Changed

- The project keeps the OSEDiff `gen + reg` training layout.
- The SD2.1 text-conditioned backbone is replaced with Ref-LDM teacher/student models.
- Dataset loading follows Ref-LDM CSV-based restoration data.
- Runtime configuration is loaded from YAML via `--config` instead of a long CLI.

## Main Entry Points

- Training: `python train_osediff_face.py --config configs/train_refldm_face.yaml`
- Inference: `python test_osediff.py --config configs/test_refldm_face.yaml`
- Timing: `python test_inference_time.py --config configs/test_refldm_face.yaml`

## Notes

- `configs/refldm_teacher.yaml` is copied from the original Ref-LDM config and remains the source of model structure truth.
- The generated student checkpoints are self-contained `.pkl` files saved under the configured `output_dir/checkpoints/`.
- The vendored Ref-LDM code lives under `refldm/`.

