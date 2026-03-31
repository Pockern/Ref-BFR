This file explains the current OSEDiff_refldm environment setup, training and inference usage, and the implemented Ref-LDM-in-OSEDiff architecture.

# OSEDiff_refldm Environment, Training, Inference, and Architecture

## 1. Overview

`OSEDiff_refldm` is a copied `OSEDiff` project that keeps the OSEDiff framework layout and the `gen + reg` training split, but replaces the original SD2.1 text-conditioned task implementation with a Ref-LDM-based one-step distillation pipeline.

Current implementation goals:

- Keep OSEDiff project organization and training loop style.
- Keep Ref-LDM latent space, teacher checkpoint, and dual conditioning.
- Replace long CLI arguments with YAML config files.
- Make the current implementation runnable from a small set of fixed entrypoints.

Main entrypoints:

- Training: `train_osediff_face.py`
- Inference: `test_osediff.py`
- Timing benchmark: `test_inference_time.py`

Main config files:

- `configs/train_refldm_face.yaml`
- `configs/test_refldm_face.yaml`
- `configs/refldm_teacher.yaml`

## 2. Directory Layout

- `configs/`
  Stores YAML configs used by the current implementation.
- `docs/`
  Stores project documentation.
- `dataloaders/refldm_dataset.py`
  Ref-LDM-style face restoration dataset adapted into the OSEDiff dataloader layout.
- `osediff.py`
  Keeps the OSEDiff wrapper class names and roles, but implements Ref-LDM teacher/student logic.
- `train_osediff_face.py`
  Current training entry.
- `test_osediff.py`
  Current one-step inference entry.
- `test_inference_time.py`
  Current timing benchmark entry.
- `refldm/`
  Vendored Ref-LDM code required by the copied OSEDiff project.

## 3. Environment Setup

### 3.1 Recommended Python Environment

The current implementation mixes copied OSEDiff code with vendored Ref-LDM code. A dedicated virtual environment or conda environment is strongly recommended.

Recommended baseline:

```bash
conda create -n osediff_refldm python=3.10 -y
conda activate osediff_refldm
```

### 3.2 Recommended Package Installation

Install OSEDiff-style base dependencies first:

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
pip install --upgrade pip
pip install -r requirements.txt
```

Then install the extra runtime packages required by the current Ref-LDM-based implementation:

```bash
pip install accelerate lpips pandas scipy opencv-python pytorch-lightning torchmetrics torchvision
```

Notes:

- `PyYAML` is already listed in `requirements.txt` and is used by `utils/config.py`.
- `lpips` is required at training time by `train_osediff_face.py`.
- `pytorch-lightning` and `torchmetrics` are required by vendored Ref-LDM modules imported through `refldm/ldm/...`.
- The current implementation added compatibility shims so `omegaconf` is no longer required for the core training/inference path.
- If you want to use GPU mixed precision, ensure your installed `torch` / `torchvision` / CUDA versions are compatible.

### 3.3 Checkpoints and Data You Need

Current default configs expect these files to exist:

- Teacher checkpoint:
  `/mnt/data/shenglong/project/checkpoints/refldm.ckpt`
- Ref-LDM VQ/VAE checkpoint:
  `/mnt/data/shenglong/project/checkpoints/vqgan.ckpt`
- Ref-LDM training CSV:
  `/mnt/data/shenglong/project/datasets/FFHQ-Ref/reference_mapping/train_references.csv`
- GT/ref image root:
  `/mnt/data/shenglong/project/datasets/ffhq-dataset/images512x512`

If your actual paths differ, update the YAML config files directly.

## 4. Config System

The current implementation no longer uses the original long command-line argument list as the main runtime interface.

Instead:

- each entry script only takes `--config`
- YAML stores the runtime values
- `utils/config.py` loads YAML into an argparse-style namespace

Example:

```bash
python train_osediff_face.py --config configs/train_refldm_face.yaml
python test_osediff.py --config configs/test_refldm_face.yaml
```

This keeps the behavior simple:

- parameter source changed from CLI to YAML
- training logic itself was not restructured around a new config framework

## 5. Training

### 5.1 Training Config

The default training config is:

- `configs/train_refldm_face.yaml`

Important fields:

- `teacher_config_path`
  Ref-LDM model structure config.
- `teacher_ckpt_path`
  Frozen multi-step teacher checkpoint.
- `student_init_ckpt_path`
  Initial checkpoint used to initialize the one-step student and reg branch.
- `vae_ckpt_path`
  Ref-LDM VQ/VAE checkpoint.
- `teacher_ddim_steps`
  Number of DDIM steps used by the frozen teacher.
- `student_timestep`
  Single timestep used by the one-step student.
- `cfg_scale`
  CFG scale for teacher sampling.
- `file_list`, `gt_dir`, `ref_dir`
  Dataset metadata and image roots.

### 5.2 Run Training

From the project root:

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python train_osediff_face.py --config configs/train_refldm_face.yaml
```

If you want to launch through `accelerate`, the current script still uses `Accelerator` internally, so you can also use:

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
accelerate launch train_osediff_face.py --config configs/train_refldm_face.yaml
```

### 5.3 Training Outputs

By default, outputs are written under:

- `experience/osediff_refldm_face/`

Important subdirectories:

- `experience/osediff_refldm_face/checkpoints/`
- `experience/osediff_refldm_face/eval/`
- `experience/osediff_refldm_face/logs/`

Student checkpoints are saved as:

- `model_<step>.pkl`

Each checkpoint stores:

- student model weights
- teacher config path
- teacher checkpoint path
- VAE checkpoint path
- student timestep

## 6. Inference

### 6.1 Inference Config

The default inference config is:

- `configs/test_refldm_face.yaml`

Important fields:

- `osediff_path`
  Path to the saved one-step student checkpoint.
- `teacher_config_path`
  Ref-LDM structure config used to reconstruct the student model.
- `teacher_ckpt_path`
  Used to build the Ref-LDM structure before loading the saved student weights.
- `vae_ckpt_path`
  Ref-LDM VQ/VAE checkpoint.
- `input_image`
  LQ input image path.
- `reference_images`
  Reference image list.
- `image_size`
  Resize target for the current pipeline.

### 6.2 Run Inference

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python test_osediff.py --config configs/test_refldm_face.yaml
```

The output image is written to:

- `output_dir/output_name`

With the current default config that means:

- `outputs/result.png`

### 6.3 Timing Benchmark

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python test_inference_time.py --config configs/test_refldm_face.yaml
```

This benchmark uses random tensors shaped like the current Ref-LDM LQ/reference inputs and measures one-step inference time.

## 7. Current Core Architecture

### 7.1 Why It Is Still “OSEDiff Framework First”

The current implementation keeps the OSEDiff outer structure:

- training still uses `OSEDiff_gen`
- training still uses `OSEDiff_reg`
- training still uses a dual-optimizer loop
- `Accelerator` orchestration is still in the OSEDiff-style entry script
- inference still goes through `OSEDiff_test`

What changed is the task-specific model internals, not the framework role split.

### 7.2 Main Components

#### `OSEDiff_gen`

Implemented in `osediff.py`.

Role in the current project:

- acts as the one-step student branch
- builds a trainable Ref-LDM model
- uses the Ref-LDM diffusion backbone as the trainable part
- keeps the latent space fixed by freezing the Ref-LDM first stage

Outputs during training:

- `student_image`
- `student_latent`
- `student_eps`
- `target_latent`
- `target_image`
- `raw_cond`
- `encoded_cond`
- `timestep`
- `noise`
- `x_t`

#### `OSEDiff_reg`

Implemented in `osediff.py`.

Role in the current project:

- keeps the OSEDiff regularization branch abstraction
- owns the frozen multi-step Ref-LDM teacher
- owns an auxiliary trainable Ref-LDM branch used for diffusion-style regularization

Main responsibilities:

- run frozen teacher DDIM sampling
- produce `teacher_latent` and `teacher_image`
- compute latent matching loss
- compute auxiliary Ref-LDM diffusion loss through `diff_loss`

#### `OSEDiff_test`

Implemented in `osediff.py`.

Role in the current project:

- keeps the OSEDiff test-wrapper role
- rebuilds the student from the saved checkpoint metadata
- runs one-step Ref-LDM inference from an LQ image plus reference images

### 7.3 Ref-LDM Conditioning in the Current Project

The current project preserves Ref-LDM’s dual conditioning design:

- `lq_image`
  Used as concat condition.
- `ref_image`
  Used as reference condition through the Ref-LDM cache-KV path.

In inference:

- the LQ image is resized and normalized to `[-1, 1]`
- all references are resized and concatenated along width
- the condition dict is passed through Ref-LDM `get_learned_conditioning`

In training:

- `RefLDMFaceDataset` returns `gt_image`, `lq_image`, `ref_image`
- `OSEDiff_gen` uses Ref-LDM `get_input()` and `get_learned_conditioning()`
- the original Ref-LDM latent and conditioning path remain the source of truth

### 7.4 Teacher and Student Flow

The current training path is:

1. Load a batch from `RefLDMFaceDataset`.
2. Use the student Ref-LDM model to encode GT and conditions.
3. Sample a single `x_t` using `student_timestep` and shared random noise.
4. Run the one-step student prediction from that `x_t`.
5. Rebuild the same `x_t` in the teacher branch.
6. Run frozen multi-step DDIM teacher sampling from that latent start.
7. Decode teacher latent to teacher image.
8. Compute the OSEDiff-style generator and regularization losses.

### 7.5 Loss Structure

The current implementation keeps the OSEDiff split into a generator loss path and a regularization loss path.

Generator-side losses:

- `loss_l2`
  MSE between student image and teacher image.
- `loss_lpips`
  LPIPS between student image and target GT image.
- `loss_kl`
  Latent matching loss between student latent and teacher latent.

Regularization-side loss:

- `loss_d`
  Auxiliary Ref-LDM diffusion training loss from the trainable reg branch.

The exact loss weights come from `configs/train_refldm_face.yaml`.

## 8. Data Flow

### 8.1 Training Data Format

`RefLDMFaceDataset` expects a CSV like Ref-LDM’s original face restoration setup.

Current required CSV columns:

- `gt_image`
- `ref_image`

Optional column if `use_given_lq: true`:

- `lq_image`

Reference handling:

- references are parsed from the CSV list field
- references can be shuffled
- references can be duplicated to `max_num_refs`
- references are concatenated along width by default

### 8.2 Degradation

If `use_given_lq: false`, LQ images are synthesized online from GT using the current `degrad_opt` section in the YAML config.

That means:

- blur
- downsample
- noise
- jpeg compression

are all defined in `configs/train_refldm_face.yaml`.

## 9. Practical Notes and Current Limitations

- The current implementation is framework-consistent, but it is still an early migration version.
- Several files copied from the original OSEDiff project are currently unused because the task has moved away from SD2.1 text-conditioning.
- The project currently relies on external checkpoints and dataset paths being correct in YAML.
- Training was not fully executed end-to-end in this environment as part of the implementation turn.
- Import-level and syntax-level checks were completed successfully on the modified entry files.

## 10. Minimal Command Summary

### Install

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
pip install -r requirements.txt
pip install accelerate lpips pandas scipy opencv-python pytorch-lightning torchmetrics torchvision
```

### Train

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python train_osediff_face.py --config configs/train_refldm_face.yaml
```

### Inference

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python test_osediff.py --config configs/test_refldm_face.yaml
```

### Benchmark

```bash
cd /mnt/data/shenglong/project/OSEDiff_refldm
python test_inference_time.py --config configs/test_refldm_face.yaml
```

