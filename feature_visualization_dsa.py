# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import VISUALIZERS
from seg.recorder import MyRecorderManager
from mmrazor.models.task_modules import ModuleOutputsRecorder, ModuleInputsRecorder, MethodOutputsRecorder
from mmrazor.visualization.local_visualizer import modify
from mmseg.visualization.local_visualizer import SegLocalVisualizer
from seg.apis import init_model, inference_model
import matplotlib.pyplot as plt
from seg.datasets.synapse import SynapseDataset
from seg.datasets.flare22 import FLARE22Dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Feature map visualization')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument('--repo', help='the corresponding repo name')
    parser.add_argument(
        '--use-norm',
        action='store_true',
        help='normalize the featmap before visualization')
    parser.add_argument(
        '--overlaid', action='store_true', help='overlaid image')
    parser.add_argument(
        '--channel-reduction',
        help='Reduce multiple channels to a single channel. The optional value'
             ' is \'squeeze_mean\', \'select_max\' or \'pixel_wise_max\'.',
        default=None)
    parser.add_argument(
        '--topk',
        type=int,
        help='If channel_reduction is not None and topk > 0, it will select '
             'topk channel to show by the sum of each channel. If topk <= 0, '
             'tensor_chw is assert to be one or three.',
        default=20)
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        help='the arrangement of featmap when channel_reduction is not None '
             'and topk > 0.',
        default=[4, 5])
    parser.add_argument(
        '--resize-shape',
        nargs='+',
        type=int,
        help='the shape to scale the feature map',
        default=None)
    parser.add_argument(
        '--alpha', help='the transparency of featmap', default=0.5)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.',
        default={})

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / (std + 1e-6)
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

def main(args):
    feat_uncertainty_maps = "feat_ours_softplus"
    add_commit = ""
    model = init_model(args.config, args.checkpoint, device=args.device)

    recorders = dict()
    recorders['uncertainty'] = dict(
        type=MethodOutputsRecorder, 
        source='seg.models.decode_heads.decode_head_edl.LossDualHead.get_uncertainty')
    recorders['conf'] = dict(
        type=MethodOutputsRecorder, 
        source='seg.models.decode_heads.decode_head_edl.LossDualHead.get_conf')
    
    # recorders['uncertainty'] = dict(
    #     type=MethodOutputsRecorder, 
    #     source='seg.models.decode_heads.decode_head_edl.LossHead.get_uncertainty')
    # recorders['conf'] = dict(
    #     type=MethodOutputsRecorder, 
    #     source='seg.models.decode_heads.decode_head_edl.LossHead.get_conf')
    # recorders['logits'] = dict(
    #     type=MethodOutputsRecorder, 
    #     source='seg.models.decode_heads.decode_head.BaseDecodeHead.predict_by_feat')
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.draw_featmap = modify
    visualizer.set_dataset_meta(classes=FLARE22Dataset.METAINFO['classes'],
                            palette=FLARE22Dataset.METAINFO['palette'],
                            dataset_name='FLARE22Dataset')

    recorder_manager = MyRecorderManager(recorders)
    recorder_manager.initialize(model)

    with recorder_manager:
        # test a single image
        result = inference_model(model, args.img, args.img.replace('img_dir', 'ann_dir').replace('jpg', 'png'))

    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None
    visualizer.add_datasample(
        name='pred.jpg', 
        image=overlaid_image,
        data_sample=result,
        out_file=f'./out_dir/{feat_uncertainty_maps}/{osp.splitext(osp.basename(args.config))[0]}/pred{add_commit}.jpg',
        withLabels=False)
    
    recorder = recorder_manager.get_recorder('uncertainty')
    # recorder = recorder_manager.get_recorder('logits')
    # record_idx = getattr(name, 'record_idx', 0)
    # data_idx = getattr(name, 'data_idx')
    feats = recorder.get_record_data()
    if isinstance(feats, torch.Tensor):
        feats = (feats,)
    # print(result.seg_logits.data)
    for i, feat in enumerate(feats):
        if args.use_norm:
            feat = norm(feat)
        print(f'{feat[0].shape}, {overlaid_image.shape}')
        drawn_img = visualizer.draw_featmap(
            feat[0],
            # result.seg_logits.data,
            overlaid_image,
            args.channel_reduction,
            topk=args.topk,
            arrangement=tuple(args.arrangement),
            resize_shape=tuple(args.resize_shape)
            if args.resize_shape else None,
            alpha=args.alpha)
        # U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
        mmcv.imwrite(mmcv.rgb2bgr(drawn_img),
                        f'./out_dir/{feat_uncertainty_maps}/{osp.splitext(osp.basename(args.config))[0]}/uncertainty{add_commit}.jpg')
        # plt.imshow(drawn_img)
        # plt.show()
    
    recorder = recorder_manager.get_recorder('conf')
    # recorder = recorder_manager.get_recorder('logits')
    # record_idx = getattr(name, 'record_idx', 0)
    # data_idx = getattr(name, 'data_idx')
    feats = recorder.get_record_data()
    if isinstance(feats, torch.Tensor):
        feats = (feats,)
    # print(result.seg_logits.data)
    for i, feat in enumerate(feats):
        if args.use_norm:
            feat = norm(feat)
        print(f'{feat[0].shape}, {overlaid_image.shape}')
        drawn_img = visualizer.draw_featmap(
            feat[0],
            # result.seg_logits.data,
            overlaid_image,
            args.channel_reduction,
            topk=args.topk,
            arrangement=tuple(args.arrangement),
            resize_shape=tuple(args.resize_shape)
            if args.resize_shape else None,
            alpha=args.alpha)
        # U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
        mmcv.imwrite(mmcv.rgb2bgr(drawn_img),
                        f'./out_dir/{feat_uncertainty_maps}/{osp.splitext(osp.basename(args.config))[0]}/conf{add_commit}.jpg')
        # plt.imshow(drawn_img)
        # plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
