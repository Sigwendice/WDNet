#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5 12:06
# @Author  : Atrio
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

from networkHelper import *
from train_hfrm import HFRM
from wave_t import DWT, IDWT
from torchvision.utils import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dwt = DWT()
idwt = IDWT()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

image_size = 32
channels = 1
batch_size = 32
img_folder = r"../datasets/ffhq/demo"
# img_folder = r'../datasets/set11'
test_data = ImageDataset(img_folder=img_folder, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

sr_list = [2, 3, 4, 5, 8]
for sr in sr_list:
    model1 = HFRM(sr=sr).to(device)
    # model1.load_state_dict(torch.load(f'models/BinA/best_sr{sr}.pth'))
    model1.load_state_dict(torch.load(f"models/A_l/A_best{sr}.pth"))
    #
    # model2 = f'./models/BinB/best{sr}.pth'
    model2 = f"./models/B_l/best{sr}.pth"
    DDPM = torch.load(model2)

    model1.eval()
    DDPM.eval()

    img_paths = [
        filename
        for filename in os.listdir(img_folder)
        if filename.endswith(("jpg", "png", "jpeg", "tif"))
    ]

    def crop_to_patches(image, crop_size=64):
        """将单张图像切割成多个patch（返回4D张量）"""
        c, h, w = image.shape
        n_patches = (h // crop_size) * (w // crop_size)
        patches = image.unfold(1, crop_size, crop_size).unfold(2, crop_size, crop_size)
        patches = patches.contiguous().view(c, n_patches, crop_size, crop_size)
        return patches.permute(1, 0, 2, 3)  # (n_patches, c, h, w)

    def merge_patches(patches, original_size=256):
        """将多个patch合并回原图（输入为4D张量）"""
        n_patches, c, h, w = patches.shape
        n_rows = original_size // h
        merged = patches.view(n_rows, n_rows, c, h, w)
        merged = merged.permute(2, 0, 3, 1, 4).contiguous()
        merged = merged.view(c, original_size, original_size)
        return merged

    with torch.no_grad():
        step = 0
        j = 0
        all_psnr = 0
        all_ssim = 0
        for inputs, _ in test_loader:

            inputs = inputs.to(device)
            full_output = torch.zeros_like(inputs)

            for i in range(inputs.shape[0]):
                print(j)
                j += 1
                # 批量处理单个图像的所有patches
                image = inputs[i]  # (C, H, W)

                # Step 1: 切割成批量patches
                patches = crop_to_patches(
                    image, crop_size=int(inputs.shape[-1] / 4)
                )  # (16, C, 64, 64)

                # Step 2: 批量处理所有patches
                l, h = dwt(patches)
                x_cond, xp_l, _ = model1(patches)

                # 拆分特征
                lh, hl, hh = torch.chunk(x_cond, chunks=3, dim=1)

                # 批量生成（关键修改点）
                samples = DDPM(
                    mode="generate",
                    image_size=int(inputs.shape[-1] / 8),
                    batch_size=len(patches),  # 使用实际patch数量作为batch_size
                    channels=channels,
                    x_start_noise=xp_l,
                )

                a = samples[-1]
                # 批量逆变换
                output_patches = idwt(samples[-1], lh, hl, hh)

                # Step 3: 合并回原图
                merged_image = merge_patches(
                    output_patches, original_size=int(inputs.shape[-1])
                )
                full_output[i] = merged_image

            # 后续评估与保存逻辑保持不变...
            for i in range(len(img_paths)):
                # x1 = (inputs[i].squeeze(0).cpu().numpy() * 255.).astype(np.uint8)
                # x2 = (full_output[i].squeeze(0).cpu().numpy() * 255.).astype(np.uint8)
                # per_psnr = psnr(x1, x2)
                # per_ssim = ssim(x1, x2, data_range=1.0)
                # print(f'PSNR: {per_psnr:.4f} dB, SSIM:{per_ssim:.4f}')
                # all_psnr += per_psnr.item()
                # all_ssim += per_ssim.item()

                # save_image(full_output[i], f"./result/test/anti_noise/10%/g_0_50/{img_paths[step]}")
                save_image(full_output[i], f"./result/test{sr}/{img_paths[step]}")
                step += 1

        # print(f'avg_psnr: {(all_psnr / step):.4f} dB')
        # print(f'avg_ssim: {(all_ssim / step):.4f}')
