# Copyright (c) wtq. All rights reserved.
import torch
import torch.nn.functional as F
import torch.nn
from .vd_fuc import expand_onehot_labels
# 求出困难样本的权重——KL
# def calculate_weighted_KL(seg_logits, dis_logits, channel_th=None):
#     # seg_logits = torch.sigmoid(seg_logits)
#     kl_distance = nn.KLDivLoss(reduction='none')
#     log_sm = torch.nn.LogSoftmax(dim=1)
#     sm = torch.nn.Softmax(dim=1)
#     log_seg_logits = log_sm(seg_logits)
#     soft_dis = sm(dis_logits)
#     if channel_th == None:
#         kl_variance = kl_distance(
#             log_seg_logits,
#             dis_logits)
#     else:
#         kl_variance = kl_distance(
#             log_seg_logits[:,channel_th:channel_th+1:,:],
#             soft_dis[:,channel_th:channel_th+1:,:])
#     variance = torch.sum(kl_variance, dim=1)
    
#     # 我们的就是要正的
#     weight = torch.exp(variance)
    
#     # 调试
#     # print('exp_variance shape',exp_variance.shape)
#     # print('Kl exp_variance mean: %.4f'%torch.mean(exp_variance[:]))
#     # print('Kl exp_variance min: %.4f'%torch.min(exp_variance[:]))
#     # print('Kl exp_variance max: %.4f'%torch.max(exp_variance[:])) 
#     return weight

def calculate_weighted_KL(pred, target, num_classes=9, ignore_index=255): 
    if (pred.shape != target.shape):
            one_hot_target = expand_onehot_labels(pred, target)
    pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
    one_hot_target = one_hot_target[:, torch.arange(num_classes) != ignore_index, :, :]  
    temperature = 1.0
    # pred = F.softmax(pred / temperature, dim=1)
    # one_hot_target = F.softmax(one_hot_target / temperature, dim=1)
    pred = pred.float()
    one_hot_target = one_hot_target.float()
    loss = F.kl_div(pred, one_hot_target, reduction='none', log_target=False)
    # loss = loss * temperature**2
    loss = torch.sum(loss, dim=1)
    weight = torch.exp(loss)
    return weight