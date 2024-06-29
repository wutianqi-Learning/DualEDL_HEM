import math
# from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Global_T(nn.Module):
    def __init__(self, eps=1e-5, init_pred_var=5.0):
        super(Global_T, self).__init__()

        self.global_T = nn.Parameter(np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()
        self.eps = eps

    def forward(self, fake_input1, fake_input2, lambda_):
        pred_var = torch.log(1.0+torch.exp(self.global_T))+self.eps
        return self.grl(pred_var, lambda_)



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