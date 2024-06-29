import math
# from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch


class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        self.mlp = InstanceTemperature()
        # self.global_T = nn.Parameter(mlp)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.mlp(fake_input1,fake_input2), lambda_)

class InstanceTemperature(nn.Module):
    def __init__(self,
                 input_dim=9):
        super(InstanceTemperature, self).__init__()
        # 连个9*2=18
        self.input_dim = input_dim * 2 
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(self.input_dim, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Softplus()
        # )
        self.mlp = nn.Sequential(
            # 第一个 3*3 => 1*3 3*1 18->64
            nn.Conv2d(in_channels=self.input_dim, 
                      out_channels=32,
                        kernel_size=(1, 3), stride=2, padding=(0, 1), 
                        bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, 
                      out_channels=32,
                        kernel_size=(3, 1), stride=2, padding=(1, 0), 
                        bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 第二个 3*3 => 1*3 3*1 32->1
            nn.Conv2d(in_channels=32, 
                      out_channels=1,
                        kernel_size=(1, 3), stride=2, padding=(0, 1), 
                        bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, 
                      out_channels=1,
                        kernel_size=(3, 1), stride=2, padding=(1, 0), 
                        bias=False),
            nn.BatchNorm2d(1),
            nn.Softplus()   
        )
        # 用全局平均池化
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self,seg_tensor, dis_tensor):
        concat_tensor = torch.cat([seg_tensor, dis_tensor], dim=1)
        temperature = self.mlp(concat_tensor)
        temperature = self.global_pooling(temperature)
        return temperature


from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)