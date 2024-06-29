# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from seg.registry import MODELS
from .epoch_decay import CosineDecay, LinearDecay
from .temp_gloabal import Global_T
from ..hem_method import get_current_consistency_weight
from ..hem_method.activate_evidence import softplus_evidence, relu_evidence, exp_evidence

@MODELS.register_module()
class MSEPulsLoss(nn.Module):
    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_mse',
                 dis_softmax: bool = False,
                 num_channels=9,
                 init_pred_var=5.0,
                 eps=1e-5,
                 max_epochs=40,
                 first_point=10,
                 second_point=20):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.dis_softmax = dis_softmax
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * \
                torch.ones(num_channels)
            )
        self.eps = eps
        self.max_epochs = max_epochs
        self.first_point = first_point
        self.second_point = second_point

    def forward(self, outputs: Tensor, dis_to_mask: Tensor, current_epoch) -> Tensor:
        if 'evidence' in self.loss_name_:
            alpha = outputs + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            outputs_soft = alpha / S
        else:
            outputs_soft = outputs.softmax(dim=1)
            
        if self.dis_softmax: 
            dis_to_mask = dis_to_mask.softmax(dim=1)
        
        mse = (dis_to_mask - outputs_soft) **2
        ramps = get_current_consistency_weight(current_epoch)
        # 分阶段过度
        a = 1.0
        if current_epoch < self.first_point:
            mse_loss = a * ramps * mse
            loss = torch.mean(mse_loss)   
            return 1.0 * loss
        else:
            if current_epoch > self.first_point and current_epoch < self.second_point:
                # 表示的是到20e 后面一项=1 ==> a=0
                a = 1.0 - current_epoch / 20
            else:
                a = 0
            gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=10)
            decay_value = gradient_decay.get_value(current_epoch)
            mlp_net = Global_T()
            mlp_net.cuda()
            mlp_net.train()
            temp = mlp_net(dis_to_mask, outputs, decay_value)
            temp = 1 + 20 * torch.sigmoid(temp)
            mse_puls = 0.5 * (mse / temp)
            mse_loss = a * ramps * mse + (1 - a) * self.loss_weight * mse_puls
            loss = torch.mean(mse_loss)
            return loss

    @property
    def loss_name(self):
        return self.loss_name_
