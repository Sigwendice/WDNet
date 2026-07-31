#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扩散模型阶段训练：HFRM 输出的小波低频 xp_l 作为条件，预测噪声。

训练完成后权重保存在 checkpoints/B_l/ 下（best{sr}.pth / epoches*_sr{sr}.pth）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from data.dataset import ImageDataset
from net.diffusion import DiffusionModel
from net.hfrm import HFRM
from net.unet import Unet
from trainers.trainer import SimpleDiffusionTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
image_size = 64
channels = 1
batch_size = 32
lambda_step = 0.1
epoches = 500
sr = 5  # 采样率，需与 HFRM 训练一致
timesteps = 1000
schedule_name = "sigmoid_beta_schedule"
lr = 1e-4

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomCrop(size=(64, 64))]
)

train_folder = config.TRAIN_FOLDER
train_data = ImageDataset(train_folder, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# 模型：冻结 HFRM，训练扩散模型
# ---------------------------------------------------------------------------
model1 = HFRM(sr=sr).to(device)
model1.load_state_dict(
    torch.load(os.path.join(config.CHECKPOINT_DIR, "BinA", "A_best5.pth"), map_location=device)
)
# 冻结 net1 的所有参数
for param in model1.parameters():
    param.requires_grad = False  # 冻结 net1 的所有参数

denoise_model = Unet(
    dim=32,
)

DDPM = DiffusionModel(
    schedule_name=schedule_name,
    timesteps=timesteps,
    beta_start=0.0001,
    beta_end=0.02,
    denoise_model=denoise_model,
).to(device)

optimizer = Adam(DDPM.parameters(), lr=lr)

Trainer = SimpleDiffusionTrainer(
    epoches=epoches,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    timesteps=timesteps,
    IFadjust_learning_rate=True,
    types="type1",
    sr=sr,
)
# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
model = Trainer(model1=model1, model2=DDPM, start_epochs=0)
