# WDNet：面向单像素成像的双分支小波扩散网络

本仓库是论文 *"Dual-branch wavelet diffusion for single-pixel imaging"*（发表于 Optics Express 34(13): 24870–24887, 2026）的官方实现。

单像素成像（SPI）是一种利用空间光调制与单点探测器、通过计算手段重建图像的技术。然而在极低采样率下，现有重建方法仍面临高频细节丢失、噪声干扰等显著挑战。

WDNet 提出一种小波域双分支重建方法：将图像在小波域分解为低频与高频分量并分别重建——高频分量由轻量级网络 HFRM 恢复纹理细节，低频分量则引导扩散过程以维持结构稳定。实验表明，在极低采样率下，WDNet 在仿真数据与真实采集系统中均能获得更清晰、更低噪的重建结果。

英文说明见 [README.md](README.md)。

## 概览

方法整体框架（小波域两阶段双路径重建，WDNet）：

![WDNet 训练流程总览](figures/training.png)

单像素成像系统原理图与实验装置：

![单像素成像系统与实验图](figures/system.png)

## 快速开始（Quick Start）

### 依赖环境

- Python 3.8+（开发与测试基于 3.12）
- 建议使用支持 CUDA 的 GPU

安装所需依赖：

```bash
pip install torch torchvision
pip install einops tqdm matplotlib numpy scipy pillow scikit-image
```

### 数据准备

代码从 `config.py` 中配置的路径加载图像。默认 `DATASET_ROOT` 指向 `../datasets/ffhq`（即项目目录的同级文件夹），目录结构如下：

```
<WDNet 的上级目录>/
└── datasets/
    └── ffhq/
        ├── train/   # 训练图像（jpg / png / jpeg / tif）
        └── demo/    # 测试图像，供 test_demo.py 使用
```

你可以按上述结构放置数据集，或直接修改 `config.py` 中的 `DATASET_ROOT`、`TRAIN_FOLDER`、`DEMO_FOLDER` 以匹配本地路径。训练时图像会被裁剪为 64×64 的 patch；测试图像在内部同样会被切分为 64×64 的 patch，因此建议使用边长为 64 整数倍（如 256×256）的方形图像。

### 训练

训练为两阶段流程。所有命令均从项目根目录运行。

**第一阶段 — HFRM（高频重建）：**

```bash
python scripts/train_hfrm.py
```

该阶段联合优化测量矩阵（AutoEncoder）与高频恢复网络。权重保存至 `checkpoints/BinA/best_sr{sr}.pth`，损失曲线保存至 `loss/BinA/`。

**第二阶段 — 扩散模型：**

```bash
python scripts/train.py
```

该阶段以 HFRM 输出的低频分量为条件训练 DDPM。需要先用第一阶段训练好的 HFRM 权重；默认会加载 `checkpoints/BinA/A_best5.pth`，因此请将第一阶段生成的 `best_sr5.pth` 重命名（或直接修改 `scripts/train.py` 中的路径）。权重保存至 `checkpoints/B_l/`。

### 测试

在演示集上运行完整重建流程（HFRM + 扩散 + 小波逆变换）：

```bash
python scripts/test_demo.py
```

结果将按 `sr_list` 中各采样率分别保存至 `result/test{sr}/`。

直接由系统仿真测量值（`.mat`）重建：

```bash
python scripts/test_from_values.py
```

该脚本依赖 `config.py` 中 `MATRIX_DIR` 配置的 `.mat` 文件。

计算真值文件夹与重建文件夹之间的 PSNR / SSIM：

```bash
python utils/metrics.py <真值目录> <重建目录>
```

## 项目结构

```
WDNet/
├── config.py                # 全局路径与通用配置（数据集、输出目录等）
├── data/
│   └── dataset.py           # 数据集类（ImageDataset、CustomMNIST）
├── net/                     # 网络模型模块
│   ├── measurement.py       # 压缩感知测量模块（AutoEncoder、BigConv、BinarizedNN）
│   ├── wave_t.py            # 小波变换（DWT/IDWT）
│   ├── hfrm.py              # HFRM 网络（支持 autoencoder / bigconv 两种测量方式）
│   ├── unet.py              # UNet 去噪骨干网络
│   └── diffusion.py         # DDPM 扩散模型
├── trainers/
│   └── trainer.py           # 训练器（TrainerBase、SimpleDiffusionTrainer、EarlyStopping）
├── utils/
│   ├── helpers.py           # 通用工具函数（embedding、upsample/downsample、extract 等）
│   ├── variance_schedule.py # 噪声方差调度（linear/cosine/quadratic/sigmoid）
│   └── metrics.py           # PSNR / SSIM 评估工具（命令行可用）
├── scripts/
│   ├── train_hfrm.py        # HFRM 阶段训练
│   ├── train.py             # 扩散模型阶段训练
│   ├── test_demo.py         # 演示测试（HFRM + 扩散模型完整重建）
│   ├── test_from_values.py  # 从 .mat 系统仿真测量值重建
│   └── value_to_conv.py     # .mat 测量值调试工具
├── figures/                 # 论文框架图（training.png、system.png）
├── checkpoints/             # 模型权重保存目录（自动创建）
├── loss/                    # 损失曲线图保存目录（自动创建）
└── result/                  # 重建结果保存目录（自动创建）
```

## 核心组件

### HFRM（高频重建模块）
位于 `net/hfrm.py`，该模块：
- 利用小波分解（DWT/IDWT）分离图像频率
- 实现注意力机制（空间注意力与通道注意力）
- 采用带归一化的残差块
- 将低频重建与高频细节增强相结合
- `measurement` 参数支持两种测量方式：
  - `"autoencoder"`：可学习测量卷积矩阵
  - `"bigconv"`：全图大矩阵测量，支持通过 `forward_from_phix` 从预计算测量值重建（测量值重建流程）

### 扩散模型
位于 `net/diffusion.py`，该组件：
- 实现去噪扩散概率模型（DDPM）
- 支持多种噪声调度（linear、cosine、sigmoid 等）
- 以 HFRM 输出的低频分量 `xp_l` 作为条件进行生成
- 包含前向与反向扩散过程

### UNet
位于 `net/unet.py`，作为去噪骨干网络：
- 为扩散步数提供时间嵌入（time embedding）
- 引入注意力机制（自注意力、交叉注意力）
- 使用带组归一化的残差块
- 包含上/下采样操作

## 模型架构

系统采用两阶段方案：
1. **HFRM 阶段**：基于小波频率分离，从压缩测量值重建图像
2. **扩散阶段**：利用去噪扩散过程进一步精炼重建结果

## 配置说明

训练超参数可在各训练脚本顶部调整：
- 图像尺寸（`image_size`）
- 批大小（`batch_size`）
- 训练轮数（`epoches`）
- 学习率（Adam 优化器内）
- 小波相关设置
- 网络维度

全局路径集中在 `config.py` 中：
- `DATASET_ROOT` / `TRAIN_FOLDER` / `DEMO_FOLDER`：数据集路径
- `CHECKPOINT_DIR` / `LOSS_DIR` / `RESULT_DIR`：输出目录
- `MATRIX_DIR`：系统仿真测量值（`.mat`）目录

## 结果

训练好的模型与日志分别保存在 `checkpoints/` 与 `loss/` 目录中。

## 致谢

本实现基于多种深度学习技术，包括：
- 去噪扩散概率模型（DDPM）
- UNet 架构
- 用于频率分析的小波变换
- 用于特征增强的注意力机制

## 引用

如果本研究对你的工作有所帮助，请引用：

```bibtex
@Article{Wen:26,
  author  = {Zhiqiang Wen and Qiu Tu and Qiurong Yan and Siying Huang and Junyuan Yin and Junli Wu and Xiaolong Luo},
  title   = {Dual-branch wavelet diffusion for single-pixel imaging},
  journal = {Opt. Express},
  volume  = {34},
  number  = {13},
  pages   = {24870--24887},
  month   = {Jun},
  year    = {2026},
  doi     = {10.1364/OE.589959}
}
```
