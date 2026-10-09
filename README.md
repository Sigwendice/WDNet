# WDNet: Dual-Branch Wavelet Diffusion for Single-Pixel Imaging

This repository is the official implementation of *"Dual-branch wavelet diffusion for single-pixel imaging"*, published in Optics Express 34(13): 24870–24887, 2026.

Single-pixel imaging (SPI) is a technique that utilizes spatial light modulation and a single-point detector to reconstruct images through computational methods. However, under ultra-low measurement rates, existing reconstruction methods still face significant challenges, such as the loss of high-frequency details and noise interference.

WDNet proposes a wavelet-domain dual-branch reconstruction method: the image is decomposed into low- and high-frequency components in the wavelet domain and reconstructed separately — the high-frequency components are restored through a lightweight network (HFRM) to recover texture details, while the low-frequency components guide a diffusion process to maintain structural stability. Experiments demonstrate that under extremely low measurement rates, WDNet achieves clearer and less noisy reconstruction results in both simulated data and real acquisition systems.

For the Chinese version, see [README-zh.md](README-zh.md).

## Overview

The overall method framework (two-stage wavelet-domain dual-path reconstruction, WDNet):

![WDNet training overview](figures/training.png)

The single-pixel imaging system schematic and the experimental setup:

![SPI system and experiment](figures/system.png)

## Quick Start

### Dependencies

- Python 3.8+ (developed and tested on 3.12)
- A CUDA-enabled GPU is recommended

Install the required packages:

```bash
pip install torch torchvision
pip install einops tqdm matplotlib numpy scipy pillow scikit-image
```

### Data Preparation

The code loads images from folders configured in `config.py`. By default `DATASET_ROOT` points to `../datasets/ffhq` (a sibling folder of the project), organized as:

```
<parent-of-WDNet>/
└── datasets/
    └── ffhq/
        ├── train/   # training images (jpg / png / jpeg / tif)
        └── demo/    # test images used by test_demo.py
```

You can either place your dataset there, or edit `DATASET_ROOT`, `TRAIN_FOLDER`, and `DEMO_FOLDER` in `config.py` to match your local paths. Training crops images into 64×64 patches; demo/test images are split into 64×64 patches internally, so use square images whose side length is a multiple of 64 (e.g., 256×256).

### Training

Training is a two-stage pipeline. Run all commands from the project root.

**Stage 1 — HFRM (high-frequency reconstruction):**

```bash
python scripts/train_hfrm.py
```

This jointly optimizes the measurement matrix (AutoEncoder) and the high-frequency recovery network. Weights are saved to `checkpoints/BinA/best_sr{sr}.pth` and loss curves to `loss/BinA/`.

**Stage 2 — Diffusion model:**

```bash
python scripts/train.py
```

This trains the DDPM using the HFRM low-frequency output as a condition. It requires the Stage-1 HFRM weights; by default it loads `checkpoints/BinA/A_best5.pth`, so rename `best_sr5.pth` produced by Stage 1 accordingly (or edit the path inside `scripts/train.py`). Weights are saved to `checkpoints/B_l/`.

### Testing

Full reconstruction pipeline (HFRM + diffusion + inverse wavelet transform) on the demo set:

```bash
python scripts/test_demo.py
```

Results are written to `result/test{sr}/` for each sampling rate in `sr_list`.

Reconstruction directly from system-simulation measurement values (`.mat`):

```bash
python scripts/test_from_values.py
```

This requires the `.mat` files configured via `MATRIX_DIR` in `config.py`.

Evaluate PSNR / SSIM between a ground-truth folder and a reconstruction folder:

```bash
python utils/metrics.py <gt_dir> <recon_dir>
```

## Project Structure

```
WDNet/
├── config.py                # Global paths and general config (datasets, output dirs, etc.)
├── data/
│   └── dataset.py           # Dataset classes (ImageDataset, CustomMNIST)
├── net/                     # Network modules
│   ├── measurement.py       # Compressed-sensing measurement module (AutoEncoder, BigConv, BinarizedNN)
│   ├── wave_t.py            # Wavelet transform (DWT/IDWT)
│   ├── hfrm.py              # HFRM network (supports 'autoencoder' / 'bigconv' measurements)
│   ├── unet.py              # UNet denoising backbone
│   └── diffusion.py         # DDPM diffusion model
├── trainers/
│   └── trainer.py           # Trainer (TrainerBase, SimpleDiffusionTrainer, EarlyStopping)
├── utils/
│   ├── helpers.py           # Utility functions (embedding, upsample/downsample, extract, etc.)
│   ├── variance_schedule.py # Noise variance schedule (linear/cosine/quadratic/sigmoid)
│   └── metrics.py           # PSNR / SSIM evaluation (CLI available)
├── scripts/
│   ├── train_hfrm.py        # Stage-1 HFRM training
│   ├── train.py             # Stage-2 diffusion training
│   ├── test_demo.py         # Demo test (full HFRM + diffusion reconstruction)
│   ├── test_from_values.py  # Reconstruct from .mat system-simulation measurements
│   └── value_to_conv.py     # .mat measurement debugging utility
├── figures/                 # Paper figures (training.png, system.png)
├── checkpoints/             # Model weights (auto-created)
├── loss/                    # Loss-curve figures (auto-created)
└── result/                  # Reconstruction outputs (auto-created)
```

## Key Components

### HFRM (High-Frequency Reconstruction Module)
Located in `net/hfrm.py`, this module:
- Uses wavelet decomposition (DWT/IDWT) to separate image frequencies
- Implements attention mechanisms (spatial and channel attention)
- Features residual blocks with normalization
- Combines low-frequency reconstruction with high-frequency detail enhancement
- The `measurement` argument supports two measurement modes:
  - `"autoencoder"`: learnable measurement convolution matrix
  - `"bigconv"`: full-image large-matrix measurement; supports reconstruction from pre-computed measurements via `forward_from_phix` (measurement-value reconstruction flow)

### Diffusion Model
Located in `net/diffusion.py`, this component:
- Implements a denoising diffusion probabilistic model (DDPM)
- Supports various noise schedules (linear, cosine, sigmoid, etc.)
- Works with the HFRM output (low-frequency component `xp_l`) for conditional generation
- Includes forward and reverse diffusion processes

### UNet
Located in `net/unet.py`, this serves as the denoising backbone:
- Features time embeddings for diffusion steps
- Incorporates attention mechanisms (self-attention, cross-attention)
- Uses residual blocks with group normalization
- Includes up/downsampling operations

## Model Architecture

The system follows a two-stage approach:
1. **HFRM Stage**: Reconstructs images from compressed measurements using wavelet-based frequency separation
2. **Diffusion Stage**: Further refines the reconstruction using a denoising diffusion process

## Configuration

Key training hyperparameters can be adjusted at the top of the training scripts:
- Image size (`image_size`)
- Batch size (`batch_size`)
- Number of epochs (`epoches`)
- Learning rate (in the Adam optimizer)
- Wavelet settings
- Network dimensions

Global paths are centralized in `config.py`:
- `DATASET_ROOT` / `TRAIN_FOLDER` / `DEMO_FOLDER`: dataset paths
- `CHECKPOINT_DIR` / `LOSS_DIR` / `RESULT_DIR`: output directories
- `MATRIX_DIR`: system-simulation measurement values (`.mat`) directory

## Results

Trained models and logs are saved in the `checkpoints/` and `loss/` directories respectively.

## Acknowledgments

This implementation builds upon various deep learning techniques including:
- Denoising Diffusion Probabilistic Models (DDPM)
- UNet architectures
- Wavelet transforms for frequency analysis
- Attention mechanisms for feature enhancement

## Citation

If you use this code in your research, please cite:

```bibtex
@Article{Wen:26,
  author  = {Zhiqiang Wen and Qiu Tu and Qiurong Yan and Siying Huang and Junyuan Yin and Junli Wu and Xiaolong Luo},
  title   = {Dual-branch wavelet diffusion for single-pixel imaging},
  journal = {Opt. Express},
  volume  = {34},
  number  = {13},
  pages   = {24870--24887},
  month   = {Jun},
  year    = {2026},
  doi     = {10.1364/OE.589959}
}
```
