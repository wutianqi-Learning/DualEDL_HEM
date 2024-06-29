import torch
import torch.nn.functional as F
from .base import BaseWeightedLoss
from ..hem_method import expand_onehot_labels
from .utils import weight_reduce_loss
from ..hem_method import get_current_consistency_weight
from .adv_utils.epoch_decay import CosineDecay
from .adv_utils.temp_gloabal import Global_T

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

class EvidenceLoss(BaseWeightedLoss):
    """Evidential MSE Loss."""
    def __init__(self, num_classes, 
                 evidence='relu', 
                 loss_type='mse', 
                 with_kldiv=True,
                 with_avuloss=False,
                 disentangle=False,
                 annealing_method='step', 
                 annealing_start=0.01, 
                 annealing_step=10,
                 loss_name='loss_evidence',
                 loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.loss_type = loss_type
        self.with_kldiv = with_kldiv
        self.with_avuloss = with_avuloss
        self.disentangle = disentangle
        self.annealing_method = annealing_method
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step
        self.eps = 1e-10
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        # self.is_binary = is_binary

    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_classes], dtype=torch.float32).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                    keepdim=True) + lnB + lnB_uni
        return kl

    def loglikelihood_loss(self, dis_alpha, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        S_dis = torch.sum(dis_alpha, dim=1, keepdim=True)
       
        loglikelihood_err = torch.sum(
           ((dis_alpha/S_dis) - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood_var += torch.sum(
            dis_alpha * (S_dis - dis_alpha) / (S_dis * S_dis * (S_dis + 1)), dim=1, keepdim=True)
        return loglikelihood_err, loglikelihood_var
    
    def adv_loglikelihood_loss(self, dis_alpha, alpha, current_epoch):
        S = torch.sum(alpha, dim=1, keepdim=True)
        S_dis = torch.sum(dis_alpha, dim=1, keepdim=True)
        # neg_log_prob = 0.5*(
        #     (pred_mean-target)**2/pred_var+torch.log(pred_var)
        #     )
        t_start = 25
        t_end = 35
        gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=10)
        decay_value = gradient_decay.get_value(current_epoch - t_start + 1)
        mlp_net = Global_T()
        mlp_net.cuda()
        mlp_net.train()
        temp = mlp_net(dis_alpha, alpha, decay_value)  # (teacher_output, student_output)
       
        temp = t_start - t_start + 1 + t_end - t_start * torch.sigmoid(temp)
        pred_var = temp.cuda()
        loglikelihood_err = torch.sum(
            0.5*(((dis_alpha/S_dis) - (alpha / S)) ** 2 / pred_var + torch.log(pred_var)), dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood_var += torch.sum(
            dis_alpha * (S_dis - dis_alpha) / (S_dis * S_dis * (S_dis + 1)), dim=1, keepdim=True)
        return loglikelihood_err, loglikelihood_var/2

    def mse_loss(self, dis, alpha, annealing_coef):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        epoch_num: the current training epoch
        """
        losses = {}
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(dis, alpha)
        losses.update({'loss_cls': loglikelihood_err, 'loss_var': loglikelihood_var})

        losses.update({'lambda': annealing_coef})
       
        return losses
    
    def adv_mse_loss(self, dis, alpha, annealing_coef, current_epoch):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        epoch_num: the current training epoch
        """
        losses = {}
        loglikelihood_err, loglikelihood_var = self.adv_loglikelihood_loss(dis, alpha, current_epoch)
        losses.update({'loss_cls': loglikelihood_err, 'loss_var': loglikelihood_var})

        losses.update({'lambda': annealing_coef})
       
        return losses


    def compute_annealing_coef(self, **kwargs):
        assert 'epoch' in kwargs, "epoch number is missing!"
        assert 'total_epoch' in kwargs, "total epoch number is missing!"
        epoch_num, total_epoch = kwargs['epoch'], kwargs['total_epoch']
        # annealing coefficient
        if self.annealing_method == 'step':
            annealing_coef = torch.min(torch.tensor(
                1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)
        else:
            raise NotImplementedError
        return annealing_coef

    def _forward(self, evidence, dis_mask, **kwargs):
        """Forward function.
        Args:
            output (torch.Tensor): The class score (before softmax).
            target (torch.Tensor): The ground truth label.
            epoch_num: The number of epochs during training.
        Returns:
            torch.Tensor: The returned EvidenceLoss loss.
        """
        
        alpha = (evidence + 1)**2
        # alpha = evidence + 1
        alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
        alpha = alpha.transpose(1, 2)  # [N, HW, C]
        alpha = alpha.contiguous().view(-1, alpha.size(2))
        # S = torch.sum(alpha, dim=1, keepdim=True)
        # evidence = alpha - 1
        # y = F.one_hot(target, num_classes=9)
        dis_mask = (dis_mask + 1)**2
        dis_mask = dis_mask.view(dis_mask.size(0), dis_mask.size(1), -1)  # [N, C, HW]
        dis_mask = dis_mask.transpose(1, 2)  # [N, HW, C]
        dis_mask = dis_mask.contiguous().view(-1, dis_mask.size(2))

        # compute annealing coefficient 
        annealing_coef = self.compute_annealing_coef(**kwargs)
        
        # compute the EDL loss
        # if kwargs['epoch'] < 25 or kwargs['epoch'] > 35:
        # if 15 < kwargs['epoch'] < 20:
        #     results = self.adv_mse_loss(dis_mask, alpha, annealing_coef, kwargs['epoch'])
        #     # if kwargs['epoch'] > 35:
        #     #     self.loss_weight = get_current_consistency_weight(kwargs['epoch']-10, kwargs['total_epoch']-10)
        #     # else:
        #     #     self.loss_weight = get_current_consistency_weight(kwargs['epoch'], kwargs['total_epoch']-10)
        # elif 35 < kwargs['epoch'] < 40:
        #     results = self.adv_mse_loss(dis_mask, alpha, annealing_coef, kwargs['epoch'])
        # else:
        #     results = self.mse_loss(dis_mask, alpha, annealing_coef)
        #     # self.loss_weight = 1.0

        results = self.mse_loss(dis_mask, alpha, annealing_coef)
        self.loss_weight = get_current_consistency_weight(kwargs['epoch'], kwargs['total_epoch'])

        return results
    
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