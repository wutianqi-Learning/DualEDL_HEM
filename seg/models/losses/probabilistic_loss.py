# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from seg.registry import MODELS
from .utils import weight_reduce_loss


def _expand_onehot_labels(target: torch.Tensor, num_classes) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    # num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


@MODELS.register_module()
class ProbabilisticLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 num_class=9,
                 ignore_index=255,
                 loss_name='loss_probabilistic'
                 ):
       
        super().__init__()
        self.loss_weight = loss_weight
        self.num_class = num_class
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                x,
                target,
                model,
                **kwargs):
        onehot_target = _expand_onehot_labels(target, self.num_class)
        # output = model(x)
        model.forward(x, onehot_target, training=True)
        elbo = model.elbo(onehot_target)
        reg_loss = l2_regularisation(model.posterior) + \
                    l2_regularisation(model.prior) + \
                    l2_regularisation(model.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
