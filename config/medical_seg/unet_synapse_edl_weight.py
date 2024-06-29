from mmengine.config import read_base
from seg.models.medical_seg import MISSFormerEDL, MISSFormer_sdf
from seg.models.medical_seg.unet import UNet
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR
from seg.models.decode_heads.decode_head_edl import LossHead
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook
from seg.models.losses.edl_weight_loss2 import EvidenceWeightLoss2
from seg.models.losses.edl_dice_distance_loss import EvidenceDiceDistance
with read_base():
    from .._base_.models.unet_r18_s4_d8_hem import *  # noqa
    from .._base_.datasets.my_synapse import *  # noqa
    from .._base_.schedules.schedule_adamw_ReduceOnPlateauLR_50epoch import *  # noqa
    from .._base_.default_runtime_epo import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
evidence_loss = dict(type=EvidenceWeightLoss2,
                      class_num=9,
                      loss_name='loss_evidence')

model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
        #  neck=None,
         decode_head=dict(num_classes=9),
         structure_type='evidence', #normal
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=UNet,
    spatial_dims=2,  # 2D
    in_channels=1,
    out_channels=9,
    kernel_size=5,
    channels=(8, 16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    )
model['decode_head'] = dict(
    type=LossHead,
    in_index=-1,
    num_classes=9,
    loss_decode=evidence_loss)

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
work_dir = '../working_synapse/res2_unet/lr_2e_4-evidence_weight_SoftPlus-CEM2'