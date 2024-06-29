
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from mmengine.visualization import Visualizer
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize

from ..hem_method import calculate_weighted_vd
from ..losses.edl_loss import relu_evidence, exp_evidence, softplus_evidence

class ProbabilisticLossHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 num_classes,
                 out_channels=None,
                 in_index=-1,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 resize_mode='bilinear',
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.resize_mode = resize_mode
        self.align_corners = align_corners

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

    def forward(self, inputs):
        """Forward function."""
        if isinstance(inputs, Tensor):
            return inputs
        else:
            output = inputs[self.in_index]
            return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType, model) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(inputs, batch_data_samples, model)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, inputs: Tensor,
                     batch_data_samples: SampleList,
                     model) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=seg_label.shape[1:] if self.resize_mode == 'trilinear' else seg_label.shape[2:],
        #     mode=self.resize_mode,
        #     align_corners=self.align_corners)

        loss = dict()

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    inputs,
                    seg_label,
                    model
                    )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    inputs,
                    seg_label,
                    weight=None,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        if len(batch_img_metas[0]['img_shape']) > 2:
            size = (batch_img_metas[0]['img_shape'][2],) + batch_img_metas[0]['img_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode=self.resize_mode,
            align_corners=self.align_corners)
        return seg_logits
    
       