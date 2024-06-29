# optimizer
from torch.optim import SGD, Adam
from mmengine.optim import ExponentialLR, LinearLR, StepLR, MultiStepLR, OptimWrapper, AmpOptimWrapper
from mmengine.optim.scheduler import PolyLR, ReduceOnPlateauLR
from mmengine.runner.loops import IterBasedTrainLoop, ValLoop, TestLoop, EpochBasedTrainLoop
from mmengine.hooks import IterTimerHook, LoggerHook, \
        ParamSchedulerHook, DistSamplerSeedHook, SyncBuffersHook,\
        RuntimeInfoHook, CheckpointHook
from mmseg.engine import SegVisualizationHook
from seg.engine import MyCheckpointHook
from mmengine.optim import CosineAnnealingLR
from torch.optim import AdamW
from mmengine.optim.scheduler import LinearLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=2e-4, weight_decay=1e-5))

# param_scheduler = [
#     dict(
#         type=ReduceOnPlateauLR,
#         monitor='mDice',
#         begin=0,
#         end=100,
#         by_epoch=True)
# ]
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=True)
]
# training schedule for 40 epochs
train_cfg = dict(
    type=EpochBasedTrainLoop, 
    # val_begin=1,
    max_epochs=100, val_interval=10)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    runtime_info=dict(type=RuntimeInfoHook),
    sync_buffers=dict(type=SyncBuffersHook),
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=100, ignore_last=False, 
                # 是否在逐次验证步骤中输出指标。
                # 当运行基于epoch的运行程序时，它可以为 "true"。
                # 如果设置为 True，after_val_epoch 
                # 将在 runner.visualizer.add_scalars 
                # 中把 step 设置为 self.epoch。
                # 否则，step 将是 self.iter。默认为 True。
                log_metric_by_epoch=True),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook, by_epoch=True, 
                    interval=1,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook, 
                       draw=True, interval=1))
