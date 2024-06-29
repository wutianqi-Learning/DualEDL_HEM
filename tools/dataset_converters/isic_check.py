import os
import mmcv
import os.path as osp
import numpy as np
from PIL import Image

mode_type = 'test'
dataset_path = f'/home/jz207/workspace/data/ISIC2018/ann_dir/{mode_type}'
print(len(os.listdir(dataset_path)))
for img_name in sorted(os.listdir(dataset_path)):
    # img = mmcv.imread(osp.join(dataset_path, img_name))
    image = Image.open(osp.join(dataset_path, img_name))
    pixel_min = image.getextrema()[0]
    pixel_max = image.getextrema()[1]

    if pixel_min >= 0 and pixel_max <= 1:
        print("Pixel values range from 0 to 1")
    elif pixel_min >= 0 and pixel_max <= 255:
        print(img_name,"Pixel values range from 0 to 255")
        break
    # assert np.all(img <= 1) and np.all(img >= 0)
print("Over!")   