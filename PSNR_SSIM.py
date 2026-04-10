#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/13 19:20
# @Author  : Atrio
from PIL import Image
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def mse(imageA, imageB):
    # 计算两个图像的均方误差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def calculate_psnr(imageA, imageB):
    # 计算两张图片的MSE
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        # MSE为0表示两张图片完全一样
        return float("inf")

    # 计算PSNR
    MAX_PIXEL = 255.0
    psnr_value = 20 * np.log10(MAX_PIXEL / np.sqrt(mse_value))
    return psnr_value


# def img_paths(folder_path1, folder_path2):
# 	image_path1 = []
# 	image_path2 = []
# 	for filename in os.listdir(folder_path1):
# 		img_paths = os.path.join(folder_path1, filename)
# 		image_path1.append(img_paths)
# 	for filename in os.listdir(folder_path2):
# 		img_paths = os.path.join(folder_path2, filename)
# 		image_path2.append(img_paths)
# 	return image_path1, image_path2


def img_paths(folder_path1, folder_path2):
    """
    获取两个文件夹中的图片路径，并按修改时间排序

    参数:
    folder_path1 (str): 第一个文件夹路径
    folder_path2 (str): 第二个文件夹路径

    返回:
    tuple: 包含两个列表的元组，每个列表包含按修改时间排序的文件路径
    """
    # 获取第一个文件夹中的所有文件及其修改时间
    image_path1 = []
    for filename in os.listdir(folder_path1):
        img_paths = os.path.join(folder_path1, filename)
        image_path1.append(img_paths)

    # 获取第二个文件夹中的所有文件及其修改时间
    files_with_mtime2 = []
    for filename in os.listdir(folder_path2):
        file_path = os.path.join(folder_path2, filename)
        if os.path.isfile(file_path):  # 确保是文件而不是目录
            mtime = os.path.getmtime(file_path)  # 获取修改时间
            files_with_mtime2.append((mtime, file_path))

    # 按修改时间排序（从最早到最新）
    files_with_mtime2.sort(key=lambda x: x[0])
    image_path2 = [path for _, path in files_with_mtime2]

    return image_path1, image_path2


def psrn(img_paths1, img_paths2):
    PSNR = 0.0
    SSIM = 0.0
    if len(img_paths1) == len(img_paths2):
        for i in range(len(img_paths1)):
            image1 = Image.open(img_paths1[i]).convert("L")
            image2 = Image.open(img_paths2[i]).convert("L")

            # 确保两张图片的尺寸相同
            assert image1.size == image2.size, "Images must be of the same size."

            # 将图片转换为numpy数组
            image1_np = np.array(image1)
            # print(image1_np)
            # print(image1_np.shape)
            image2_np = np.array(image2)
            # print(image2.size)
            # psnr = calculate_psnr(image1_np, image2_np)
            psnr_value = psnr(image1_np, image2_np)
            # 假设 image1 和 image2 是 numpy 数组
            ssim_value, _ = ssim(image1_np, image2_np, full=True)

            print(f"PSNR: {psnr_value:.2f} dB, SSIM:{ssim_value:.4f}")
            PSNR += psnr_value
            SSIM += ssim_value
    print(f"avg_psnr:{PSNR/len(img_paths1):.4f}")
    print(f"avg_ssim:{SSIM/len(img_paths1):.4f}")


folder1 = r"../datasets/ffhq/demo"
# folder2 = r'.\result\test8'
# folder2 = r'C:\Users\Atrio\Desktop\result\WGDM\Llinear+AB\64\test8'

folder2 = r"C:\Users\Atrio\Desktop\result\WGDM\Llinear+AB\PN"
# folder2 = r'./result/H_g/test8'
# folder2 = r'C:\Users\Atrio\Desktop\result\WGDM\Conv+A+B_h\256\test8'
# folder2 = r'D:\M\Code\1\DDPM\MPIGAN_Code\MPIGAN_Code\result'


# folder1 = r'../datasets/BrainImages_test'
# folder2 = r'./result/mri/test8'

images1, images2 = img_paths(folder1, folder2)
# print(images1)
# print(images2)
psrn(images1, images2)
