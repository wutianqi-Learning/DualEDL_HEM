# Copyright (c) wtq. All rights reserved.
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from .activate_evidence import relu_evidence, exp_evidence, softplus_evidence

def custom_function_cos(logits):
    mask = logits != 0  # 创建一个掩码，表示非零元素的位置
    result = torch.zeros_like(logits)  # 创建一个与 logits 形状相同的全零张量
    result[mask] = torch.cos(math.pi * logits[mask] / 2)  # 对非零元素进行计算
    return result

def custom_function_sin(logits):
    mask = logits != 0  # 创建一个掩码，表示非零元素的位置
    result = torch.zeros_like(logits)  # 创建一个与 logits 形状相同的全零张量
    result[mask] = torch.sin(math.pi * logits[mask] / 2)  # 对非零元素进行计算
    return result

def custom_function_cos_easy(logits):
    mask = logits != 0  # 创建一个掩码，表示非零元素的位置
    result = torch.zeros_like(logits)  # 创建一个与 logits 形状相同的全零张量
    result[mask] = torch.cos(math.pi * logits[mask] / 2 - 0.5)  # 对非零元素进行计算
    return result

# seg_label one-hot
def expand_onehot_labels(pred: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target

# 得到假阳、假阴、容易
def parts(seg_logits, seg_label, threshold):
    # 阈值化
    thresholded_tensor = (seg_logits >= threshold).to(seg_logits) 
    # 交集操作,得到容易点
    easy_bool = thresholded_tensor.bool() & seg_label.bool()
    # 并集操作
    all_bool = thresholded_tensor.bool() | seg_label.bool()
    # 异或操作得到困难点
    difficult_bool = easy_bool ^ all_bool
    # 困难的与分割图求交集,得到假阳性
    false_positive_bool = difficult_bool & thresholded_tensor.bool()
    # 困难的与金标准求交集,得到假阴性
    false_negative_bool = difficult_bool & seg_label.bool()
    
    false_positive_logits = false_positive_bool * seg_logits
    false_negative_logits = false_negative_bool * seg_logits
    easy_logits = easy_bool * seg_logits
    return false_positive_logits, false_negative_logits, easy_logits

# 三个部分加权和
def calculate_weighted(false_positive_logits, false_negative_logits, easy_logits):
    weight = custom_function_sin(false_positive_logits) + \
             custom_function_cos(false_negative_logits) + \
             custom_function_cos(easy_logits)
    return weight


def multi_channel_vd(seg_logits: torch.Tensor, seg_label: torch.Tensor, 
                     threshold, ignore_index=255):

    # 先把label转成one-hot处理 
    if (seg_logits.shape != seg_label.shape):
            one_hot_target = expand_onehot_labels(seg_logits, seg_label)
    num_classes = seg_logits.size(1)
    seg_logits = seg_logits[:, torch.arange(num_classes) != ignore_index, :, :]
    one_hot_target = one_hot_target[:, torch.arange(num_classes) != ignore_index, :, :]  
    # 然后是每一个channel 做韦恩图 计算权重
    weights = 0
    weight_list = []
    for channel_th in range(0,seg_logits.size(1)):
        temp_seg_logits = seg_logits[:,channel_th:channel_th+1,:,:]
        temp_seg_logits = temp_seg_logits.squeeze(1)
        temp_seg_target = one_hot_target[:,channel_th:channel_th+1,:,:]
        temp_seg_target = temp_seg_target.squeeze(1)
        # 得到三个区域
        false_positive_logits, false_negative_logits, easy_logits = \
            parts(temp_seg_logits, temp_seg_target, threshold)
        weight_list.append(calculate_weighted(false_positive_logits, false_negative_logits, easy_logits))
         # Stack the tensors in the tensor_list along a new dimension
    for weight_i in weight_list:
        weights += weight_i
    # weight_size = weight.size()
    # # 将 N * H * W 维度展平为 N * (H * W)
    # weight_flattened = weight.view(weight_size[0], -1)
    # # 在 N * (H * W) 维度上应用 Softmax 函数
    # softmax = nn.Softmax(dim=1)
    # weight_flattened = softmax(weight_flattened)
    # # 将展平后的结果还原为 N * H * W 维度
    # weight_tensor = weight_flattened.view(weight_size)
    # weight = weight_tensor + 1
    return weights * 10
    
# 求出困难样本的权重 map
# 正常就是大于0.5是预测对的
def calculate_weighted_vd(alpha, seg_label, threshold=0.5):
    weight = None
    if threshold == None:
        threshold=0.5
    # alpha = evidence + 1

    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    weight = multi_channel_vd(prob, seg_label, threshold)
    # weight = weight / weight.min()

    return weight

