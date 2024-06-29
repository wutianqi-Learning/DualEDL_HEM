from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """
        ret = self._forward(*args, **kwargs)
        loss = 0.0
        if 'weight' not in kwargs:
            weight = None
        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    # if k == 'loss_cls':
                    #     # if weight is specified, apply element-wise weight
                    #     if weight is not None:
                    #         assert weight.dim() == ret[k].dim()
                    #         if weight.dim() > 1:
                    #             assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
                    #         ret[k] = ret[k] * weight
                    # ret[k] = ret[k].mean()
                    loss += ret[k]
        else:
            ret *= self.loss_weight
        loss *= self.loss_weight
        return loss