#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5 12:06
# @Author  : Atrio
import os
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        """
        初始化时加载文件夹中的所有图片路径
        :param img_folder: 存放图片的文件夹路径
        :param transform: 对图片进行的变换操作（如数据增强、归一化等）
        """
        self.img_folder = img_folder
        self.transform = transform
        # 获取文件夹中所有的图片文件
        self.img_paths = [
            os.path.join(img_folder, filename)
            for filename in os.listdir(img_folder)
            if filename.endswith(("jpg", "png", "jpeg", "tif"))
        ]

    def __len__(self):
        # 返回图片的数量
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.img_paths[idx]
        # 打开图片
        image = Image.open(img_path)

        # 如果有transform操作，则对图片进行处理
        if self.transform:
            image = self.transform(image)

        return image, image


class CustomMNIST(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, _ = super(CustomMNIST, self).__getitem__(index)
        return img, img
