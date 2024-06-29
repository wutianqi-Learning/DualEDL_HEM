from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend
from mmseg.visualization.local_visualizer import SegLocalVisualizer
from seg.models.segmentors.seg_tta import SegTTAModel
# default_scope = 'mmseg'
default_scope = None
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends=[dict(type=LocalVisBackend)]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type=SegTTAModel)
