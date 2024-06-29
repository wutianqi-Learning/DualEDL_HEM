# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..hem_method import get_current_consistency_weight
from seg.registry import MODELS


@MODELS.register_module()
class MSELoss(nn.Module):

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_mse',
                 dis_softmax: bool = False):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.dis_softmax = dis_softmax

    def forward(self, outputs: Tensor, dis_to_mask: Tensor, current_epoch) -> Tensor:
        outputs_soft = outputs.softmax(dim=1)
        if self.dis_softmax: 
            dis_to_mask = dis_to_mask.softmax(dim=1)
        ramps = get_current_consistency_weight(current_epoch)
        loss = torch.mean((dis_to_mask - outputs_soft) ** 2 * ramps)

        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_
