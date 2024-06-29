# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from seg.datasets.dataset_wrappers import MultiImageMixDataset
from seg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ConvertPixel(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        img_seg = results['gt_seg_map']
        img_seg[img_seg == 0] = 0
        img_seg[img_seg == 255] = 1
        return results
    
@TRANSFORMS.register_module()
class BioMedical3DPad(BaseTransform):
    """Pad the biomedical 3d image & biomedical 3d semantic segmentation maps.

    Required Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Modified Keys:

    - img (np.ndarry): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.

    Added Keys:

    - pad_shape (Tuple[int, int, int]): The padded shape.

    Args:
        pad_shape (Tuple[int, int, int]): Fixed padding size.
            Expected padding shape (Z, Y, X).
        pad_val (float): Padding value for biomedical image.
            The padding mode is set to "constant". The value
            to be filled in padding area. Default: 0.
        seg_pad_val (int): Padding value for biomedical 3d semantic
            segmentation maps. The padding mode is set to "constant".
            The value to be filled in padding area. Default: 0.
    """

    def __init__(self,
                 pad_shape: Tuple[int, int, int],
                 pad_val: float = 0.,
                 seg_pad_val: int = 0) -> None:

        # check pad_shape
        assert pad_shape is not None
        if not isinstance(pad_shape, tuple):
            assert len(pad_shape) == 3

        self.pad_shape = pad_shape
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.pad_shape``

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: The dict contains the padded image and shape
                information.
        """
        padded_img = self._to_pad(
            results['img'], pad_shape=self.pad_shape, pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape[1:]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to ``self.pad_shape`` if
        ``gt_seg_map`` is not None in results dict.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Update the padded gt seg map in dict.
        """
        if results.get('gt_seg_map', None) is not None:
            pad_gt_seg = self._to_pad(
                results['gt_seg_map'][None, ...],
                pad_shape=results['pad_shape'],
                pad_val=self.seg_pad_val)
            results['gt_seg_map'] = pad_gt_seg[0]

    @staticmethod
    def _to_pad(img: np.ndarray,
                pad_shape: Tuple[int, int, int],
                pad_val: Union[int, float] = 0) -> np.ndarray:
        """Pad the given 3d image to a certain shape with specified padding
        value.

        Args:
            img (ndarray): Biomedical image with shape (N, Z, Y, X)
                to be padded. N is the number of modalities.
            pad_shape (Tuple[int,int,int]): Expected padding shape (Z, Y, X).
            pad_val (float, int): Values to be filled in padding areas
                and the padding_mode is set to 'constant'. Default: 0.

        Returns:
            ndarray: The padded image.
        """
        # compute pad width
        d = max(pad_shape[0] - img.shape[1], 0)
        pad_d = (d // 2, d - d // 2)
        h = max(pad_shape[1] - img.shape[2], 0)
        pad_h = (h // 2, h - h // 2)
        w = max(pad_shape[2] - img.shape[2], 0)
        pad_w = (w // 2, w - w // 2)

        pad_list = [(0, 0), pad_d, pad_h, pad_w]

        img = np.pad(img, pad_list, mode='constant', constant_values=pad_val)
        return img

    def transform(self, results: dict) -> dict:
        """Call function to pad images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'pad_shape={self.pad_shape}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'seg_pad_val={self.seg_pad_val})'
        return repr_str

@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - img_norm_cfg

      - mean
      - std
      - to_rgb


    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB before
            normlizing the image. If ``to_rgb=True``, the order of mean and std
            should be RGB. If ``to_rgb=False``, the order of mean and std
            should be the same order of the image. Defaults to True.
    """

    def __init__(self,
                 grayscale: bool = False,
                 to_rgb: bool = True) -> None:
        self.grayscale = grayscale
        if grayscale:
            self.to_rgb = False
        else:
            self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """Function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key 'img_norm_cfg' key is added in to
            result dict.
        """
        img = results['img']
        if self.grayscale:
            mean = img.mean()
            std = img.std()
        else:
            mean = [img[..., i].mean() for i in range(img.shape[-1])]
            std = [img[..., i].std() for i in range(img.shape[-1])]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        results['img'] = mmcv.imnormalize(img, mean, std,
                                          self.to_rgb)
        results['img'] -= results['img'].min()
        results['img_norm_cfg'] = dict(
            mean=mean, std=std, to_rgb=self.to_rgb)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_rgb={self.to_rgb})'
        return repr_str

class BioMedicalRandomGamma(BaseTransform):
    """Using random gamma correction to process the biomedical image.

    Modified from
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/color_transforms.py#L132 # noqa:E501
    With licence: Apache 2.0

    Required Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X),
        N is the number of modalities, and data type is float32.

    Modified Keys:
    - img

    Args:
        prob (float): The probability to perform this transform. Default: 0.5.
        gamma_range (Tuple[float]): Range of gamma values. Default: (0.5, 2).
        invert_image (bool): Whether invert the image before applying gamma
            augmentation. Default: False.
        per_channel (bool): Whether perform the transform each channel
            individually. Default: False
        retain_stats (bool): Gamma transformation will alter the mean and std
            of the data in the patch. If retain_stats=True, the data will be
            transformed to match the mean and standard deviation before gamma
            augmentation. Default: False.
    """

    def __init__(self,
                 prob: float = 0.5,
                 gamma_range: Tuple[float] = (0.5, 2),
                 invert_image: bool = False,
                 per_channel: bool = False,
                 retain_stats: bool = False):
        assert 0 <= prob and prob <= 1
        assert isinstance(gamma_range, tuple) and len(gamma_range) == 2
        assert isinstance(invert_image, bool)
        assert isinstance(per_channel, bool)
        assert isinstance(retain_stats, bool)
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    @cache_randomness
    def _do_gamma(self):
        """Whether do adjust gamma for image."""
        return np.random.rand() < self.prob

    def _adjust_gamma(self, img: np.array):
        """Gamma adjustment for image.

        Args:
            img (np.array): Input image before gamma adjust.

        Returns:
            np.arrays: Image after gamma adjust.
        """

        if self.invert_image:
            img = -img

        def _do_adjust(img):
            if retain_stats_here:
                img_mean = img.mean()
                img_std = img.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(self.gamma_range[0], 1), self.gamma_range[1])
            img_min = img.min()
            img_range = img.max() - img_min  # range
            img = np.power(((img - img_min) / float(img_range + 1e-7)),
                           gamma) * img_range + img_min
            if retain_stats_here:
                img = img - img.mean()
                img = img / (img.std() + 1e-8) * img_std
                img = img + img_mean
            return img

        retain_stats_here = self.retain_stats
        if not self.per_channel:
            img = _do_adjust(img)
        else:
            for c in range(img.shape[0]):
                img[c] = _do_adjust(img[c])
        if self.invert_image:
            img = -img
        return img

    def transform(self, results: dict) -> dict:
        """Call function to perform random gamma correction
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with random gamma correction performed.
        """
        do_gamma = self._do_gamma()

        if do_gamma:
            results['img'] = self._adjust_gamma(results['img'])
        else:
            pass
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'gamma_range={self.gamma_range},'
        repr_str += f'invert_image={self.invert_image},'
        repr_str += f'per_channel={self.per_channel},'
        repr_str += f'retain_stats={self.retain_stats}'
        return repr_str