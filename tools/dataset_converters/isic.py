# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import tempfile
import zipfile
import numpy as np
import mmcv
from mmengine.utils import mkdir_or_exist

CHASE_DB1_LEN = 28 * 3
TRAINING_LEN = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CHASE_DB1 dataset to mmsegmentation format')
    parser.add_argument('--dataset_path',
                        default='/home/jz207/workspace/data/ISIC2018/ann_dir_0_255/train',
                        type=str,
                        help='path of CHASEDB1.zip')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('/home/jz207/workspace/data/ISIC2018/')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    mode_type = 'train'
    dataset_path = f'/home/jz207/workspace/data/ISIC2018/ann_dir_0_255/{mode_type}'
    # 读取文件
    print(len(os.listdir(dataset_path)))
    for img_name in sorted(os.listdir(dataset_path)):
        img = mmcv.imread(osp.join(dataset_path, img_name))
        img = img[:, :, 0] // 128
        assert np.all(img <= 1) and np.all(img >= 0)
        # The annotation img should be divided by 128, because some of
            # the annotation imgs are not standard. We should set a
            # threshold to convert the nonstandard annotation imgs. The
            # value divided by 128 is equivalent to '1 if value >= 128
            # else 0'
        if osp.splitext(img_name)[1] == '.png':
            mmcv.imwrite(
                img,
                osp.join(out_dir, 'ann_dir', mode_type, img_name))
            
    print('validation files process finished...')
    mode_type = 'val'
    dataset_path = f'/home/jz207/workspace/data/ISIC2018/ann_dir_0_255/{mode_type}'
    # 读取文件
    print(len(os.listdir(dataset_path)))
    for img_name in sorted(os.listdir(dataset_path)):
        img = mmcv.imread(osp.join(dataset_path, img_name))
        img = img[:, :, 0] // 128
        assert np.all(img <= 1) and np.all(img >= 0)
        # The annotation img should be divided by 128, because some of
            # the annotation imgs are not standard. We should set a
            # threshold to convert the nonstandard annotation imgs. The
            # value divided by 128 is equivalent to '1 if value >= 128
            # else 0'
        if osp.splitext(img_name)[1] == '.png':
            mmcv.imwrite(
                img,
                osp.join(out_dir, 'ann_dir', mode_type, img_name))
            
    print('validation files process finished...')
    
    mode_type = 'test'
    dataset_path = f'/home/jz207/workspace/data/ISIC2018/ann_dir_0_255/{mode_type}'
    # 读取文件
    print(len(os.listdir(dataset_path)))
    for img_name in sorted(os.listdir(dataset_path)):
        img = mmcv.imread(osp.join(dataset_path, img_name))
        img = img[:, :, 0] // 128
        assert np.all(img <= 1) and np.all(img >= 0)
        # The annotation img should be divided by 128, because some of
            # the annotation imgs are not standard. We should set a
            # threshold to convert the nonstandard annotation imgs. The
            # value divided by 128 is equivalent to '1 if value >= 128
            # else 0'
        if osp.splitext(img_name)[1] == '.png':
            mmcv.imwrite(
                img[:, :, 0] // 128,
                osp.join(out_dir, 'ann_dir', mode_type, img_name))
    print('test files process finished...')

    print('all files process finished...')
    print('Done!')


if __name__ == '__main__':
    main()
