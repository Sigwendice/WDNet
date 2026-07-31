#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/16 10:05
# @Author  : Atrio
"""HFRM（高频重建模块）：压缩感知测量值 -> 初始重建 -> 小波频域分离 -> 高频恢复。

measurement 参数可选：
    - "autoencoder": 可学习测量矩阵（对应原 train_hfrm.py 的 AutoEncoder 流程）
    - "bigconv": 全图大矩阵测量（对应原 value_model.py 的 BigConv 流程，支持
      从系统仿真得到的测量值 forward_from_phix 直接重建）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.wave_t import DWT, IDWT
from net.measurement import AutoEncoder, BigConv


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class AvgPool2d(nn.Module):
    """自适应平均池化的固定分辨率替换实现（推理时保持池化尺寸稳定）。"""

    def __init__(
        self,
        kernel_size=None,
        base_size=None,
        auto_pad=True,
        fast_imp=False,
        train_size=None,
    ):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return "kernel_size={}, base_size={}, stride={}, fast_imp={}".format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(
                    w - 1, self.kernel_size[1] // r2
                )
                out = (
                    s[:, :, :-k1, :-k2]
                    - s[:, :, :-k1, k2:]
                    - s[:, :, k1:, :-k2]
                    + s[:, :, k1:, k2:]
                ) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = (
                s[:, :, :-k1, :-k2],
                s[:, :, :-k1, k2:],
                s[:, :, k1:, :-k2],
                s[:, :, k1:, k2:],
            )
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode="replicate")

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    """递归替换模型中的 AdaptiveAvgPool2d 为 AvgPool2d。"""
    for n, m in model.named_children():
        if len(list(m.children())) > 0:

            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(
                base_size=base_size, fast_imp=fast_imp, train_size=train_size
            )
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base:
    """将模型的 AdaptiveAvgPool2d 替换为固定核尺寸实现，并预热一次前向。"""

    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class SpatialAttn(nn.Module):
    """空间注意力：通道前一半与后一半逐元素相乘。"""

    def __init__(self, mid_dim):
        super(SpatialAttn, self).__init__()
        self.mid_dim = mid_dim

    def forward(self, x):
        x1 = x[:, 0 : self.mid_dim, :, :]

        x2 = x[:, self.mid_dim :, :, :]
        return x1 * x2


class ChannelAttn(nn.Module):
    """通道注意力：全局平均池化 + 1x1 卷积生成通道权重。"""

    def __init__(self, chan_dim):
        super(ChannelAttn, self).__init__()

        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.chan_conv = nn.Conv2d(
            in_channels=chan_dim,
            out_channels=chan_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        input = x
        x = self.pool2d(x)
        x = self.chan_conv(x)

        return input * x


class ResidualBlock(nn.Module):
    """带空间/通道注意力的残差块。"""

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dim * 2,
            out_channels=dim * 2,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dim * 2,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.spatial_attn = SpatialAttn(mid_dim=dim)
        self.channel_attn = ChannelAttn(chan_dim=dim)

        self.conv4 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim * 2,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        input = x

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.spatial_attn(x)
        x = self.channel_attn(x)

        x = self.conv3(x)

        y = input + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.spatial_attn(x)
        x = self.conv5(x)

        y = y + x * self.gamma
        return y


class HFRM(nn.Module):
    """HFRM: 测量值初始重建 + 小波频域分离 + 高频子带恢复。

    注意：in_channel 指 conv_in 的输入通道数，即 DWT 输出的高频子带通道数
    （对单通道图像为 3 = LH+HL+HH）。

    forward 返回 (高频残差, 低频估计, 初始重建)：
        - x_cond: 恢复后的高频子带（用于扩散模型的逆变换合成）
        - xp_l: 初始重建的低频分量（作为扩散模型的条件）
        - xp: 初始重建图像
    """

    def __init__(
        self,
        in_channel=3,
        dim=32,
        mid_blk_num=6,
        enc_blk_nums=[2, 2, 2, 2],
        dec_blk_nums=[2, 2, 2, 2],
        sr=5,
        measurement="autoencoder",
    ):
        super(HFRM, self).__init__()
        self.sr = sr
        self.measurement = measurement
        if measurement == "autoencoder":
            # 可学习测量矩阵（与网络联合训练）
            self.pr = AutoEncoder(sr=self.sr)
        elif measurement == "bigconv":
            # 全图大矩阵测量，支持从预计算的测量值重建
            self.pr = BigConv(sr=self.sr)
        else:
            raise ValueError(
                "measurement 参数必须为 {autoencoder} 或 {bigconv}".format(
                    autoencoder="autoencoder", bigconv="bigconv"
                )
            )
        self.dwt = DWT()
        self.idwt = IDWT()

        self.conv_in = nn.Conv2d(
            in_channels=in_channel,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mid_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[ResidualBlock(dim) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(dim, 2 * dim, 2, 2))
            dim = dim * 2

        self.mid_blks = nn.Sequential(*[ResidualBlock(dim) for _ in range(mid_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            dim = dim // 2
            self.decoders.append(
                nn.Sequential(*[ResidualBlock(dim) for _ in range(num)])
            )

        self.conv_out = nn.Conv2d(
            in_channels=dim,
            out_channels=in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        """x: 传入原始图片，进行初始重建 + 按高频恢复联合优化。"""
        xp = self.pr(x)
        return self._process_after_xp(xp)

    def forward_from_phix(self, phix):
        """新流程：输入为系统仿真预计算的测量值 phix。"""
        xp = self.pr.forward_from_phix(phix)
        return self._process_after_xp(xp)

    def _process_after_xp(self, xp):
        # 共享的后处理步骤：小波分解 -> 编码器-解码器恢复高频子带
        xp_l, h = self.dwt(xp)
        input = h
        x = self.conv_in(h)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.mid_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.conv_out(x)
        x = x + input
        return x, xp_l, xp


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("autoencoder", "bigconv"):
        net = HFRM(sr=5, measurement=name).to(device)
        x = torch.randn(size=(32, 1, 64, 64)).to(device)
        out = net(x)
        print(f"measurement={name}, output shapes:", [o.shape for o in out])
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params_count = sum([torch.numel(p) for p in model_parameters])
        print("Total number of parameters: ", params_count)
