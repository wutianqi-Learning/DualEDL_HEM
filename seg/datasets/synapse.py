# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import os
import os.path as osp
import mmengine
from seg.registry import DATASETS
# from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets import SynapseDataset


# @DATASETS.register_module()
# class SynapseDataset(BaseSegDataset):
#     """Synapse dataset.

#     Before dataset preprocess of Synapse, there are total 13 categories of
#     foreground which does not include background. After preprocessing, 8
#     foreground categories are kept while the other 5 foreground categories are
#     handled as background. The ``img_suffix`` is fixed to '.jpg' and
#     ``seg_map_suffix`` is fixed to '.png'.
#     """
#     METAINFO = dict(
#         classes=('background', 'aorta', 'gallbladder', 'left_kidney',
#                  'right_kidney', 'liver', 'pancreas', 'spleen', 'stomach'),
#         palette=[[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0],
#                  [0, 255, 255], [255, 0, 255], [255, 255, 0], [60, 255, 255],
#                  [240, 240, 240]])

#     def __init__(self,
#                  img_suffix='.jpg',
#                  seg_map_suffix='.png',
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

@DATASETS.register_module()
class SynapseDataset(SynapseDataset):
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if hasattr(self, 'case_list'):
            lines = self.case_list
        else:
            if osp.isfile(self.ann_file):
                lines = mmengine.list_from_file(
                    self.ann_file, backend_args=self.backend_args)
            else:
                lines = os.listdir(img_dir)

        case_nums = dict()

        lines.sort()

        for line in lines:
            case_name = line.strip()
            imgs = os.listdir(osp.join(img_dir, case_name))
            imgs.sort()
            case_nums[case_name] = len(imgs)
            for img in imgs:
                data_info = dict(img_path=osp.join(img_dir, case_name, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, case_name, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []

                data_info['case_name'] = case_name
                data_list.append(data_info)

        self._metainfo.update(case_nums=case_nums)
        if self._indices is not None and self._indices > 0:
            return data_list[:self._indices]
        else:
            return data_list

# @DATASETS.register_module()
# class SynapseDataset_npz(SynapseDataset):
#     def load_data_list(self) -> List[dict]:
#         """Load annotation from directory or annotation file.

#         Returns:
#             list[dict]: All data info of dataset.
#         """
#         data_list = []
#         img_dir = self.data_prefix.get('img_path', None)
#         if osp.isfile(self.ann_file):
#             lines = mmengine.list_from_file(
#                 self.ann_file, backend_args=self.backend_args)
#             for line in lines:
#                 img_name = line.strip()
#                 data_info = dict(
#                     img_path=osp.join(img_dir, img_name + self.img_suffix))
#                 if ann_dir is not None:
#                     seg_map = img_name + self.seg_map_suffix
#                     data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
#                 data_info['label_map'] = self.label_map
#                 data_info['reduce_zero_label'] = self.reduce_zero_label
#                 data_info['seg_fields'] = []
#                 data_list.append(data_info)
#         else:
#             for img in fileio.list_dir_or_file(
#                     dir_path=img_dir,
#                     list_dir=False,
#                     suffix=self.img_suffix,
#                     recursive=True,
#                     backend_args=self.backend_args):
#                 data_info = dict(img_path=osp.join(img_dir, img))
#                 if ann_dir is not None:
#                     seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
#                     data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
#                 data_info['label_map'] = self.label_map
#                 data_info['reduce_zero_label'] = self.reduce_zero_label
#                 data_info['seg_fields'] = []
#                 data_list.append(data_info)
#             data_list = sorted(data_list, key=lambda x: x['img_path'])
#         return data_list
