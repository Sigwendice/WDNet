#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 11:13
# @Author  : Atrio
"""测试：从系统仿真得到的测量值 (.mat) 直接重建。

流程：加载 .mat 测量值 -> HFRM(measurement="bigconv") 重建 + 高频恢复
      -> 扩散模型精化低频 -> IDWT 合成最终图像。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io

import config
from net.hfrm import HFRM
from net.wave_t import DWT, IDWT
from torchvision.utils import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_value(sr, mat_path):
    """读取 .mat 文件中的测量值，reshape 为卷积输入形状 [1, sr^2*16, 1, 1]。"""
    data = scipy.io.loadmat(mat_path)
    array = data["Y_test"]
    M, _ = array.shape

    final_tensor = torch.from_numpy(array) / 100.0
    final_tensor = final_tensor.reshape(1, sr * sr * 16, 1, 1)
    return final_tensor


dwt = DWT()
idwt = IDWT()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

image_size = 32
channels = 1

sr_list = [2, 3, 4, 5, 8]
mr = config.MEASUREMENT_RATES  # 测量率(采样率百分比)，与 sr_list 一一对应

save_dir = os.path.join(config.RESULT_DIR, "system")
os.makedirs(save_dir, exist_ok=True)

for sr, rate in zip(sr_list, mr):
    mat_path = os.path.join(config.MATRIX_DIR, f"hua_y{rate}.mat")
    values = load_value(sr=sr, mat_path=mat_path)

    model1 = HFRM(sr=sr, measurement="bigconv", in_channel=3).to(device)
    model1.load_state_dict(
        torch.load(
            os.path.join(config.CHECKPOINT_DIR, "F_BIN", "A_best{}.pth".format(sr)),
            map_location=device,
        )
    )

    DDPM = torch.load(os.path.join(config.CHECKPOINT_DIR, "F_BIN", "B_best{}.pth".format(sr)))

    model1.eval()
    DDPM.eval()

    with torch.no_grad():

        values = values.to(device)
        output, xp_l, xp = model1.forward_from_phix(values)
        # 拆分特征
        lh, hl, hh = torch.chunk(output, chunks=3, dim=1)

        # 批量生成（关键修改点）
        samples = DDPM(
            mode="generate",
            image_size=image_size,
            batch_size=1,  # 使用实际patch数量作为batch_size
            channels=channels,
            x_start_noise=xp_l,
        )

        # 批量逆变换
        output_patches = idwt(samples[-1], lh, hl, hh)

        save_image(output_patches, os.path.join(save_dir, f"hua{rate}.png"))
