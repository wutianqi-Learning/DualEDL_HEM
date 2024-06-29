import torch
import numpy as np
from torch import nn

class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha, device, is_split_batch=True):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self._device = device
        self.is_split_batch = is_split_batch
        self.is_evaluation = False

    def set_evaluation(self, is_evaluation):
        """
        评估时可以(也可以不管)将is_evaluation设为True,训练时设为False,这样就类似于直接计算CEL
        :param is_evaluation: bool
        :return:
        """
        self.is_evaluation = is_evaluation

    def _g2bin(self, g, bin):
        return torch.floor(g * (bin - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def use_alpha(self, bin_count):
        if (self._alpha != 0):
            if (self.is_evaluation):
                if (self._last_bin_count == None):
                    self._last_bin_count = bin_count
                else:
                    bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
                    self._last_bin_count = bin_count
        return bin_count

    def forward(self, x, target):
        """
        :param x: torch.Tensor,[B,C,*]
        :param target: torch.Tensor,[B,*]
        :return: loss
        """
        g = torch.abs(self._custom_loss_grad(x, target)).detach()
        weight = torch.zeros((x.size(0), x.size(2), x.size(3)))
        if self.is_split_batch:
            #是否对每个batch分开统计梯度,我实验时发现分开统计loss会更容易收敛,可能因为模型中用了batch normalization？
            N = x.size(2) * x.size(3)
            bin = (int)(N // self._bins)
            bin_idx = self._g2bin(g, bin)
            bin_idx = torch.clamp(bin_idx, max=bin - 1)
            bin_count = torch.zeros((x.size(0), bin))
            for i in range(x.size(0)):
                bin_count[i] = torch.from_numpy(np.bincount(torch.flatten(bin_idx[i].cpu()), minlength=bin))
                bin_count[i] *= (bin_count[i] > 0).sum()

            bin_count = self.use_alpha(bin_count)
            gd = torch.clamp(bin_count, min=1)
            beta = N * 1.0 / gd
            for i in range(x.size(0)):
                weight[i] = beta[i][bin_idx[i]]
        else:
            N = x.size(0) * x.size(2) * x.size(3)
            bin = (int)(N // self._bins)
            bin_idx = self._g2bin(g, bin)
            bin_idx = torch.clamp(bin_idx, max=bin - 1)
            bin_count = torch.from_numpy(np.bincount(torch.flatten(bin_idx.cpu()), minlength=bin))
            bin_count *= (bin_count > 0).sum()

            bin_count = self.use_alpha(bin_count)
            gd = torch.clamp(bin_count, min=1)
            beta = N * 1.0 / gd
            weight = beta[bin_idx]

        return self._custom_loss(x, target, weight)


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha, device, num_classes, ignore_classes=None, class_weights=None, is_split_batch=True):
        """
        :param bins: int 不是bin,这里将取数据[B,C,X,Y]的size计算bin=[B*]X*Y,B不一定乘
        :param alpha: float。
        :param device:
        :param num_classes: int。分类数量。
        :param ignore_classes: [int]。不计算的
        :param class_weights: torch.Tensor,每个类型的权重
        :param is_split_batch: bool,是否分离batch统计
        """
        super(GHMC_Loss, self).__init__(bins, alpha, device, is_split_batch)
        self.num_classes = num_classes
        self.ignore_classes = ignore_classes
        self.class_weights = class_weights

    def _custom_loss(self, x, target, weight):
        """
        计算loss
        :param x: torch.Tensor,[B,C,*]
        :param target: torch.Tensor,[B,*]
        :param weight: torch.Tensor,[B,C,*]
        :return: loss
        """
        if (self.is_evaluation):
            return torch.mean(
                (torch.nn.NLLLoss(weight=self.class_weights, reduction='none')(
                    torch.log_softmax(x, 1), target)))
        else:
            return torch.mean(
                (torch.nn.NLLLoss(weight=self.class_weights, reduction='none')(
                    torch.log_softmax(x, 1), target)).mul(weight.to(self._device).detach()))

    def _custom_loss_grad(self, x, target):
        """
        统计梯度
        :param x: torch.Tensor,[B,C,*]
        :param target: torch.Tensor,[B,*]
        :return: 梯度信息
        """
        g = (torch.softmax(x, 1).detach() - make_one_hot(target.unsqueeze(1), self.num_classes).to(self._device)). \
            gather(1, target.unsqueeze(1)).squeeze(1)
        if self.ignore_classes != None:
            a = torch.tensor(0.0, dtype=torch.float32).to(self._device)
            for class_id in self.ignore_classes:
                g = torch.where(target != class_id, g, a)
        return g

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    # input=torch.squeeze(input,dim=-1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result
