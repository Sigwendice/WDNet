#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 10:37
# @Author  : Atrio
import scipy.io
import numpy as np

# 读取 .mat 文件
data = scipy.io.loadmat(r"C:\Users\Atrio\Desktop\6.25.mat")  # 替换为你的 .mat 文件路径
array = data["Y_test"]  # 替换为你的数组在 .mat 文件中的键名

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
print(final_tensor)
print("最终张量的形状:", final_tensor.shape)
