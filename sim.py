#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/14 12:23
# @Author  : Atrio
import torch
from torch import nn
import torch.nn.functional as F


class BinarizedNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# M_rate match
# i
# 1        0.00390625     ×
# 2        0.015625       √       0.016
# 3        0.03515625     √       0.035
# 4        0.0625         √       0.0625
# 5        0.09765625     √       0.1
# 6        0.0140625      ×
# 7        0.019140625    ×
# 8        0.25           √       0.25
# 9        0.3164025      ×
class AutoEncoder(nn.Module):  # 构造自编码器
    def __init__(self, sr=None):
        super(AutoEncoder, self).__init__()
        self.sr = sr  # 设置采样率
        self.lambda_step = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

        self.conv1_w = nn.Parameter(
            torch.randn(self.sr**2, 1, 16, 16), requires_grad=True
        )  # 随机测量矩阵 高分辨率64*64
        self.recov_conv1_w = nn.Parameter(torch.randn(256, 1, self.sr, self.sr))
        self.recov_conv1_b = nn.Parameter(
            torch.randn(
                256,
            )
        )
        nn.init.xavier_normal_(self.conv1_w)
        nn.init.xavier_normal_(self.recov_conv1_w)
        # self.conv1_w = nn.Parameter(BinarizedNN.apply(self.conv1_w))
        # self.conv1_w = BinarizedNN.apply(self.conv1_w)

        self.relu = nn.ReLU()

        self.pixelshuffle1 = nn.PixelShuffle(self.sr)
        self.pixelshuffle2 = nn.PixelShuffle(16)

        self.unflatten = nn.Unflatten(1, (1, 64, 64))

    def forward(self, x):
        # self.conv1_w = nn.Parameter(BinarizedNN.apply(self.conv1_w)) * 0.01
        conv1 = F.conv2d(x, self.conv1_w, stride=16)  # [batch_size, sr**2, 4, 4]

        # poisson_lam = 0.05
        # poisson_noise = (torch.poisson(torch.ones(conv1.shape)) * poisson_lam).to(device)
        gaussian_std = 0
        gaussian_mean = 50
        # 生成高斯噪声 (独立控制均值和方差)
        gaussian_noise = (gaussian_std * torch.randn(conv1.shape) + gaussian_mean).to(
            device
        )
        # # 添加噪声到测量值
        noisy_conv1 = conv1 + gaussian_noise / 255.0

        pixel1 = self.pixelshuffle1(noisy_conv1)  # [batch_size,1,4*sr, 4*sr]
        recov_conv1 = F.conv2d(
            pixel1, self.recov_conv1_w, bias=self.recov_conv1_b, stride=self.sr
        )  # [batch_size,256,4,4]
        pixel2 = self.pixelshuffle2(recov_conv1)  # [batch_size,1,64,64]

        x = pixel2 - self.lambda_step * (
            F.conv_transpose2d(
                (F.conv2d(pixel2, self.conv1_w, stride=16) - conv1),
                self.conv1_w,
                stride=16,
            )
        )

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"


class BigConv(nn.Module):  # 构造自编码器
    def __init__(self, sr=None):
        super(BigConv, self).__init__()
        self.sr = sr  # 设置采样率

        self.Phi = nn.Parameter(
            nn.init.xavier_normal_(torch.Tensor((self.sr**2) * 16, 4096))
        )

    def forward(self, x):
        # Phi = self.Phi
        Phi = nn.Parameter(BinarizedNN.apply(self.Phi)) * 0.01
        print(Phi)
        Phiweight = Phi.contiguous().view((self.sr**2) * 16, 1, 64, 64).to(device)
        Phix = F.conv2d(
            x, Phiweight, padding=0, stride=64, bias=None
        )  # [batch_size, sr**2, 1, 1]

        PhiTweight = (
            self.Phi.t().contiguous().view(4096, (self.sr**2) * 16, 1, 1).to(device)
        )
        PhiTb = F.conv2d(Phix, PhiTweight, padding=0, bias=None)
        xp = torch.nn.PixelShuffle(64)(PhiTb)
        return xp


if __name__ == "__main__":
    net = BigConv(sr=5)
    x = torch.randn([32, 1, 64, 64]).to(device)
    print(net(x).shape)
