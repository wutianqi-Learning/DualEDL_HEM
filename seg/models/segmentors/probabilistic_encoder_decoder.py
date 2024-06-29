# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from seg.registry import MODELS
from seg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
from ..utils import resize
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


def model_PU(x, model):

    y = model.sample_m(x, m=8, testing=True)
    return y

@MODELS.register_module()
class ProbabilisticEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    # def __init__(self,
    #              backbone: ConfigType,
    #              decode_head: ConfigType,
    #              neck: OptConfigType = None,
    #              auxiliary_head: OptConfigType = None,
    #              train_cfg: OptConfigType = None,
    #              test_cfg: OptConfigType = None,
    #              data_preprocessor: OptConfigType = None,
    #              pretrained: Optional[str] = None,
    #              init_cfg: OptMultiConfig = None,
    #              structure_type='normal'):
        # super().__init__(
        #     data_preprocessor=data_preprocessor, 
        #     init_cfg=init_cfg)
        # if pretrained is not None:
        #     assert backbone.get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone.pretrained = pretrained
        # self.backbone = MODELS.build(backbone)
        # if neck is not None:
        #     self.neck = MODELS.build(neck)
        # self._init_decode_head(decode_head)
        # self._init_auxiliary_head(auxiliary_head)

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        # self.structure_type = structure_type

        # assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # x = self.backbone(inputs)
        x = model_PU(inputs, self.backbone)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg, self.backbone)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x = self.extract_feat(inputs)
        x = inputs
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses