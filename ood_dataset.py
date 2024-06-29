import os
import cv2
import numpy as np
from mmseg.datasets.transforms.transforms import BioMedicalGaussianNoise

root_dir = '数据集根目录的路径'  # 替换为您的数据集根目录的路径
folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

gaussian_noise = BioMedicalGaussianNoise()

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    file_names = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
        image = gaussian_noise(image)  # 应用高斯噪声变换
        cv2.imwrite(file_path, image)  # 保存图像