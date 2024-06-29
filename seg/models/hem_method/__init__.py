# Copyright (c) wtq. All rights reserved.
from .kl_fuc import calculate_weighted_KL
from .vd_fuc import calculate_weighted_vd
from .vd_fuc import expand_onehot_labels
from .current_weight import get_current_consistency_weight
from .activate_evidence import relu_evidence, exp_evidence, softplus_evidence
__all__= ['calculate_weighted_KL', 'calculate_weighted_vd', 
          'expand_onehot_labels', 'get_current_consistency_weight',
          'relu_evidence', 'exp_evidence', 'softplus_evidence']