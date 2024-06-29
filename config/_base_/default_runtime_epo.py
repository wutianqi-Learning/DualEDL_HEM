from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend
from mmseg.visualization import SegLocalVisualizer
from seg.models.segmentors.seg_tta import SegTTAModel
from mmengine.runner.log_processor import LogProcessor
# default_scope = 'mmseg'
default_scope = None
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=True)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(type=SegLocalVisualizer, 
                  vis_backends=vis_backends,
                  name='visualizer')

log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type=SegTTAModel)