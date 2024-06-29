# optimizer
from torch.optim import SGD, Adam
from mmengine.optim import ExponentialLR, LinearLR, StepLR, MultiStepLR, OptimWrapper
from mmengine.optim.scheduler import PolyLR
from mmengine.runner.loops import IterBasedTrainLoop, ValLoop, TestLoop, EpochBasedTrainLoop
from mmengine.hooks import IterTimerHook, LoggerHook, \
        ParamSchedulerHook, DistSamplerSeedHook, SyncBuffersHook,\
        RuntimeInfoHook, CheckpointHook
from mmseg.engine import SegVisualizationHook
from seg.engine import MyCheckpointHook

param_scheduler = [
    # dict(type=LinearLR, start_factor=0.01, by_epoch=True, begin=0, end=10),
    # dict(
    #     type=MultiStepLR,
    #     begin=0,
    #     end=40,
    #     by_epoch=True,
    #     milestones=[15, 30],
    #     gamma=0.1)
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40,
        by_epoch=True)
]

# optimizer
optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    clip_grad=dict(max_norm=40, norm_type=2))
# training schedule for 40 epochs
train_cfg = dict(
    type=EpochBasedTrainLoop, 
    max_epochs=40, val_begin=5, val_interval=1)
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
