# HFRM-Diffusion Image Reconstruction Project

This project implements a high-frequency reconstruction module (HFRM) combined with a diffusion model for image processing tasks. The system uses wavelet decomposition to separate image frequencies and applies deep learning techniques for reconstruction.

## Project Structure

```
├── dataset.py              # Custom dataset classes for image loading
├── diffusion.py            # Diffusion model implementation
├── networkHelper.py        # Helper functions and modules
├── PSNR_SSIM.py            # Evaluation metrics (PSNR, SSIM)
├── sim.py                  # Simulation modules (AutoEncoder, BigConv)
├── test_demo.py            # Demonstration/testing scripts
├── test_from_values.py     # Testing from saved values
├── train.py                # Main training script for diffusion model
├── trainNetworkHelper.py   # Training helper utilities
├── train_hfrm.py           # HFRM training implementation
├── unet.py                 # UNet architecture for denoising
├── value_model.py          # Value modeling components
├── value_to_conv.py        # Value to convolution conversion
├── varianceSchedule.py     # Noise variance scheduling
└── wave_t.py               # Wavelet transform implementations
```

## Key Components

### HFRM (High-Frequency Reconstruction Module)
Located in `train_hfrm.py`, this module:
- Uses wavelet decomposition (DWT/IDWT) to separate image frequencies
- Implements attention mechanisms (spatial and channel attention)
- Features residual blocks with normalization
- Combines low-frequency reconstruction with high-frequency detail enhancement

### Diffusion Model
Located in `diffusion.py`, this component:
- Implements a denoising diffusion probabilistic model (DDPM)
- Supports various noise schedules (linear, cosine, sigmoid, etc.)
- Works with the HFRM output for conditional generation
- Includes forward and reverse diffusion processes

### UNet
Located in `unet.py`, this serves as the denoising backbone:
- Features time embeddings for diffusion steps
- Incorporates attention mechanisms (self-attention, cross-attention)
- Uses residual blocks with group normalization
- Includes up/downsampling operations

## Installation

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install einops tqdm
   ```

## Usage

### Training HFRM
```bash
python train_hfrm.py
```

### Training Diffusion Model
```bash
python train.py
```

Note: Ensure you have the dataset available at `../datasets/ffhq/train` or modify the path in `train.py`.

## Model Architecture

The system follows a two-stage approach:
1. **HFRM Stage**: Reconstructs images from compressed measurements using wavelet-based frequency separation
2. **Diffusion Stage**: Further refines the reconstruction using a denoising diffusion process

## Configuration

Key parameters can be adjusted in the training scripts:
- Image size (`image_size`)
- Batch size (`batch_size`)
- Number of epochs (`epoches`)
- Learning rate (in Adam optimizer)
- Wavelet settings
- Network dimensions

## Results

Trained models and logs are saved in the `models/` and `loss/` directories respectively.

## Acknowledgments

This implementation builds upon various deep learning techniques including:
- Denoising Diffusion Probabilistic Models (DDPM)
- UNet architectures
- Wavelet transforms for frequency analysis
- Attention mechanisms for feature enhancement
