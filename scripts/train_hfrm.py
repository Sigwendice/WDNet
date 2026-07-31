#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/16 10:05
# @Author  : Atrio
"""HFRM 阶段训练：联合优化测量矩阵(AutoEncoder)与高频恢复网络。

损失 = 高频子带 MSE + 重建图像 MSE（小波逆变换回空间域比较）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from data.dataset import ImageDataset
from net.hfrm import HFRM
from net.wave_t import DWT, IDWT

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
image_size = 64
batch_size = 32
epoches = 500
sr = 5  # 采样率
lr = 1e-4

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomCrop(size=(image_size, image_size))]
)

train_data = ImageDataset(config.TRAIN_FOLDER, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# 模型与优化器
# ---------------------------------------------------------------------------
dwt = DWT()
idwt = IDWT()
net = HFRM(sr=sr, measurement="autoencoder").to(device)
optimizer = Adam(net.parameters(), lr=lr)

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
save_dir = os.path.join(config.CHECKPOINT_DIR, "BinA")
loss_sub_dir = os.path.join(config.LOSS_DIR, "BinA")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(loss_sub_dir, exist_ok=True)

mean_loss = []
min_loss = float("inf")
for i in range(epoches):
    losses = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (features, _) in loop:  # 用于高分辨率图片64*64
        input = features.to(device)
        l, h = dwt(input)

        output_h, xp_l, x_p = net(input)

        loss1 = F.mse_loss(output_h, h)

        lh, hl, hh = torch.chunk(output_h, chunks=3, dim=1)
        xp = idwt(xp_l, lh, hl, hh)
        loss2 = F.mse_loss(xp, input)

        loss3 = F.mse_loss(xp_l, l)
        # print('loss1:{},loss2:{}'.format(loss1, loss2))
        loss = loss1 + loss2
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新信息
        loop.set_description(f"Epoch [{i}/{epoches}]")
        loop.set_postfix(loss=loss.item())

    mean_loss.append(losses / len(train_loader))

    if min_loss > losses:
        torch.save(
            net.state_dict(),
            os.path.join(save_dir, "best_sr{}.pth".format(sr)),
        )
        min_loss = losses

    if i == (epoches - 1):
        torch.save(
            net.state_dict(),
            os.path.join(save_dir, "epoches{}_sr{}.pth".format(i + 1, sr)),
        )

# ---------------------------------------------------------------------------
# 损失曲线
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(1, epoches + 1), mean_loss)
plt.title(f"H_Loss-{sr} vs. Epoches")
plt.xlabel("Epochs")
plt.ylabel("H_Loss")
plt.grid(True)
plt.savefig(os.path.join(loss_sub_dir, f"epoches{epoches}_sr{sr}.png"))
plt.close()
