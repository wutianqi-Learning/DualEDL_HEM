# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from seg.registry import MODELS
from .decode_head_dualhem_edl import BaseDecodeHeadHEMEDL
from ..hem_method.activate_evidence import *


# 已修改为回归分支带有Conv的双分支结构
@MODELS.register_module()
class FCNHeadConvHEMEDL(BaseDecodeHeadHEMEDL):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 evidence='softplus',
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.evidence = evidence
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        # # 1x1 卷积
        # self.conv1x1 = ConvModule(
        #         self.out_channels,
        #         self.out_channels,
        #         kernel_size=1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        self.conv1x3x1 = nn.Sequential(
            # 1*3 
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            # 3*1
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        # 添加一个 tanh
        self.tanh = nn.Tanh()

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    # 改头部输出分割分支和回归分支
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        # dis_output = output.clone().to(output.device)
        # dis_output[:, 1:] = self.tanh(dis_output[:, 1:])
        # dis_output = self.conv1x1(output)
       
        # get evidence
        if self.evidence == 'relu':
            evidence = relu_evidence(output)
        elif self.evidence == 'exp':
            evidence = exp_evidence(output)
        elif self.evidence == 'softplus':
            evidence = softplus_evidence(output)
        else:
            raise NotImplementedError
        dis_output = self.conv1x3x1(output)
        dis_output = self.tanh(dis_output)
        # res_dis = dis_output + evidence
        return evidence, dis_output