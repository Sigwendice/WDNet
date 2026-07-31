#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/16 10:05
# @Author  : Atrio
import torch
from torch import nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class DWT(nn.Module):
    def __init__(self, wavelet="haar"):
        super(DWT, self).__init__()
        self.wavelet = wavelet
        # 预定义 Haar 小波滤波器（可扩展为其他小波）
        self.ll_filter = (torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0).to(device)
        self.lh_filter = (torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0).to(device)
        self.hl_filter = (torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0).to(device)
        self.hh_filter = (torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0).to(device)

    def forward(self, x):
        # 输入 x: [B_h, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)  # 合并批次和通道维度

        # 使用卷积实现小波分解
        ll = F.conv2d(x, self.ll_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        lh = F.conv2d(x, self.lh_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        hl = F.conv2d(x, self.hl_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        hh = F.conv2d(x, self.hh_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)

        # 分离通道维度
        ll = ll.reshape(B, C, H // 2, W // 2)
        lh = lh.reshape(B, C, H // 2, W // 2)
        hl = hl.reshape(B, C, H // 2, W // 2)
        hh = hh.reshape(B, C, H // 2, W // 2)
        return ll, torch.cat([lh, hl, hh], dim=1)
        # return ll, (lh, hl, hh)


class IDWT(nn.Module):
    def __init__(self, wavelet="haar"):
        super(IDWT, self).__init__()
        self.wavelet = wavelet
        # 转置滤波器用于逆变换
        self.ll_filter = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        self.lh_filter = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
        self.hl_filter = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0
        self.hh_filter = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0

    def forward(self, ll, lh, hl, hh):
        # 输入 ll: [B_h, C, H, W], 高频分量同理
        B, C, H, W = ll.shape
        x = torch.zeros(B, C, H * 2, W * 2, device=ll.device)

        # 逆变换：转置卷积
        x[:, :, 0::2, 0::2] = (
            ll * self.ll_filter[0, 0]
            + lh * self.lh_filter[0, 0]
            + hl * self.hl_filter[0, 0]
            + hh * self.hh_filter[0, 0]
        )
        x[:, :, 0::2, 1::2] = (
            ll * self.ll_filter[0, 1]
            + lh * self.lh_filter[0, 1]
            + hl * self.hl_filter[0, 1]
            + hh * self.hh_filter[0, 1]
        )
        x[:, :, 1::2, 0::2] = (
            ll * self.ll_filter[1, 0]
            + lh * self.lh_filter[1, 0]
            + hl * self.hl_filter[1, 0]
            + hh * self.hh_filter[1, 0]
        )
        x[:, :, 1::2, 1::2] = (
            ll * self.ll_filter[1, 1]
            + lh * self.lh_filter[1, 1]
            + hl * self.hl_filter[1, 1]
            + hh * self.hh_filter[1, 1]
        )
        return x


class DWT1(nn.Module):
    def __init__(self, wavelet="haar"):
        super(DWT1, self).__init__()
        self.wavelet = wavelet
        # 预定义 Haar 小波滤波器（可扩展为其他小波）
        self.ll_filter = (torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2).to(device)
        self.lh_filter = (torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2).to(device)
        self.hl_filter = (torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2).to(device)
        self.hh_filter = (torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2).to(device)

    def forward(self, x):
        # 输入 x: [B_h, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)  # 合并批次和通道维度

        # 使用卷积实现小波分解
        ll = F.conv2d(x, self.ll_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        lh = F.conv2d(x, self.lh_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        hl = F.conv2d(x, self.hl_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)
        hh = F.conv2d(x, self.hh_filter.unsqueeze(0).unsqueeze(0), stride=2, padding=0)

        # 分离通道维度
        ll = ll.reshape(B, C, H // 2, W // 2)
        lh = lh.reshape(B, C, H // 2, W // 2)
        hl = hl.reshape(B, C, H // 2, W // 2)
        hh = hh.reshape(B, C, H // 2, W // 2)
        # return ll, torch.cat([lh, hl, hh], dim=1)
        return ll, (lh, hl, hh)


class IDWT1(nn.Module):
    def __init__(self, wavelet="haar"):
        super(IDWT1, self).__init__()
        self.wavelet = wavelet
        # 转置滤波器用于逆变换
        self.ll_filter = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / math.sqrt(2)
        self.lh_filter = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / math.sqrt(2)
        self.hl_filter = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / math.sqrt(2)
        self.hh_filter = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / math.sqrt(2)

    def forward(self, ll, lh, hl, hh):
        # 输入 ll: [B_h, C, H, W], 高频分量同理
        B, C, H, W = ll.shape
        x = torch.zeros(B, C, H * 2, W * 2, device=ll.device)

        # 逆变换：转置卷积
        x[:, :, 0::2, 0::2] = (
            ll * self.ll_filter[0, 0]
            + lh * self.lh_filter[0, 0]
            + hl * self.hl_filter[0, 0]
            + hh * self.hh_filter[0, 0]
        )
        x[:, :, 0::2, 1::2] = (
            ll * self.ll_filter[0, 1]
            + lh * self.lh_filter[0, 1]
            + hl * self.hl_filter[0, 1]
            + hh * self.hh_filter[0, 1]
        )
        x[:, :, 1::2, 0::2] = (
            ll * self.ll_filter[1, 0]
            + lh * self.lh_filter[1, 0]
            + hl * self.hl_filter[1, 0]
            + hh * self.hh_filter[1, 0]
        )
        x[:, :, 1::2, 1::2] = (
            ll * self.ll_filter[1, 1]
            + lh * self.lh_filter[1, 1]
            + hl * self.hl_filter[1, 1]
            + hh * self.hh_filter[1, 1]
        )
        return x


if __name__ == "__main__":
    net1 = DWT()
    net2 = IDWT()
    x = torch.randn([32, 1, 64, 64])
    ll, h = net1(x)
    lh, hl, hh = torch.chunk(h, chunks=3, dim=1)
    x1 = net2(ll, lh, hl, hh)
    for name, param in net2.named_parameters():
        print(f"{name}")
