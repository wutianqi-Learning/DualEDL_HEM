from mmengine.config import read_base
from seg.models.medical_seg import MISSFormerEDL, MISSFormer_sdf
from seg.models.medical_seg.unet import UNet_sdf
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from seg.models.decode_heads.decode_head_edl import LossDualHead
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook
from seg.models.losses.edl_weight_loss2 import EvidenceWeightLoss2
from seg.models.losses.edl_dice_distance_loss import EvidenceDiceDistance
from seg.models.losses.mse_edl_loss import EvidenceLoss as MSE_EDL_loss
from seg.models.losses.dice_edl_loss import DiceDELDisLoss
with read_base():
    from .._base_.models.unet_r18_s4_d8_hem import *  # noqa
    from .._base_.datasets.my_synapse import *  # noqa
    from .._base_.schedules.schedule_adamw_ReduceOnPlateauLR_50epoch import *  # noqa
    from .._base_.default_runtime_epo import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
evidence_loss_edl = dict(type=EvidenceWeightLoss2,
                      class_num=9,
                      loss_name='loss_evidence')
evidence_loss_edl_dis = dict(type=EvidenceDiceDistance,
                      class_num=9,
                      loss_name='loss_distance_dice')
evidence_loss_mse = dict(type=MSE_EDL_loss,
                      num_classes=9,
                      with_kldiv=False,
                      with_avuloss=False,
                      annealing_method='exp',
                      loss_name='loss_mse')
# evidence_loss = dict(type=EvidenceLoss,
#                       num_classes=9,
#                       evidence='softplus',
#                       loss_type='digamma',  # cross_entropy
#                       with_kldiv=True,
#                       with_avuloss=True,
#                       annealing_method='exp',
#                       loss_name='loss_evidence')
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
        #  neck=None,
         decode_head=dict(num_classes=9),
         structure_type='evidence', #normal
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=UNet_sdf,
    spatial_dims=2,  # 2D
    in_channels=1,
    out_channels=9,
    kernel_size=5,
    channels=(8, 16, 32, 64,128),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    )
model['decode_head'] = dict(
    type=LossDualHead,
    num_classes=9,
    loss_decode=[evidence_loss_edl, evidence_loss_mse, 
                 evidence_loss_edl_dis
                #  dict(type=DiceDELDisLoss, loss_name='loss_dis_dice', loss_weight=0.5)
                 ])

# custom hooks
custom_hooks = [dict(type=SetEpochInfoHook)]
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='resunet-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
work_dir = '../working_synapse/res2_unet/lr_0002-ReduceOnPlateauLR_50epoch-evidence-AdvMSE-daul_weight_softlplus-relu_CEM2-ML'
