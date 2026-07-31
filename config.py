#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局路径与通用配置。所有脚本统一从这里读取路径，避免硬编码。"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 数据集路径
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "datasets", "ffhq"))
TRAIN_FOLDER = os.path.join(DATASET_ROOT, "train")  # 训练集
DEMO_FOLDER = os.path.join(DATASET_ROOT, "demo")  # 测试集（演示用）

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # 模型权重保存目录
LOSS_DIR = os.path.join(PROJECT_ROOT, "loss")  # 损失曲线图保存目录
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")  # 重建结果保存目录

# 系统仿真测量值 (.mat) 所在目录，如 E:\实验代码\matrix\hua\hua_y10.mat
MATRIX_DIR = r"E:\实验代码\matrix\hua"

# ---------------------------------------------------------------------------
# 通用配置
# ---------------------------------------------------------------------------
# 采样率 sr 与测量率(采样率百分比)的对应关系，test_from_values.py 中使用
MEASUREMENT_RATES = ["1_6", "3_5", "6_25", "10", "25"]


def ensure_dirs():
    """确保所有输出目录存在。"""
    for d in (CHECKPOINT_DIR, LOSS_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
