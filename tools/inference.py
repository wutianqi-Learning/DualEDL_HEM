# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from seg.utils import register_all_modules
from seg.apis import MMSegInferencer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_dir', help='image file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def find_best_in_multi_run(args):
    dirs = [dir for dir in os.listdir(args.checkpoint) if dir.startswith('run')]
    dirs = [osp.join(args.checkpoint, dir) for dir in dirs]
    dirs = [dir for dir in dirs if osp.isdir(dir)]
    max = 0
    max_ckpts = None
    for dir in dirs:
        ckpts = [file for file in os.listdir(dir) if file.startswith('best')]
        ckpts.sort()
        metric = float(ckpts[0][11:13]) + float(ckpts[0][14:16]) * 0.01
        if max < metric:
            max = metric
            max_ckpts = osp.join(dir, ckpts[0])
    assert max_ckpts is not None

    return max_ckpts

def main():
    args = parse_args()

    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)
    # dirs = [dir for dir in os.listdir(args.checkpoint) if dir.startswith('run')]
    # dirs = [osp.join(args.checkpoint, dir) for dir in dirs]
    # dirs = [dir for dir in dirs if osp.isdir(dir)]
    # max = 0
    # max_ckpts = None
    # for dir in dirs:
    #     ckpts = [file for file in os.listdir(dir) if file.startswith('best')]
    #     ckpts.sort()
    #     metric = float(ckpts[0][11:13]) + float(ckpts[0][14:16]) * 0.01
    #     if max < metric:
    #         max = metric
    #         max_ckpts = osp.join(dir, ckpts[0])
    # assert max_ckpts is not None
    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 将模型加载到内存中
    inferencer = MMSegInferencer(model=cfg,
                                 weights=args.checkpoint)
    out_dir = osp.join('./out_dir', osp.splitext(osp.basename(args.config))[0])
    mkdir_or_exist(out_dir)
    # 推理
    inferencer(args.img_dir,
               out_dir=out_dir,
               show=False)

if __name__ == '__main__':
    main()