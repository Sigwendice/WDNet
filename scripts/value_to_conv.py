#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 10:37
# @Author  : Atrio
"""调试工具：将 .mat 测量值文件 (Y_test) 读取并打印重排后的形状。

用法: python scripts/value_to_conv.py <mat文件路径>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.io
import numpy as np

import config


def load_value_conv(mat_path):
    """读取 .mat 文件中的 Y_test，重排为 (num_blocks, 1, 4, 4)。"""
    data = scipy.io.loadmat(mat_path)
    array = data["Y_test"]

    # 确保数组的形状为 (M, 1)，其中 M 是 16 的倍数
    M, _ = array.shape
    assert M % 16 == 0, "M 必须是 16 的倍数"

    # 计算新的维度
    num_blocks = M // 16

    # 重塑数组为 (num_blocks, 16)
    reshaped_array = array.reshape(num_blocks, 16)

    # 将每个 16 元素的块转换为 4x4 矩阵
    blocks = reshaped_array.reshape(num_blocks, 4, 4)

    # 添加额外的维度以形成最终的形状为 (num_blocks, 1, 4, 4)
    final_tensor = blocks.reshape(num_blocks, 1, 4, 4)
    return final_tensor


if __name__ == "__main__":
    default_path = os.path.join(config.MATRIX_DIR, "hua_y6_25.mat")
    mat_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    final_tensor = load_value_conv(mat_path)
    print(final_tensor)
    print("最终张量的形状:", final_tensor.shape)
