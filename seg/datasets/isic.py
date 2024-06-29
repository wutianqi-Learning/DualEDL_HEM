import mmengine.fileio as fileio
from seg.registry import DATASETS
from .basesegdataset import BaseSegDataset


# @DATASETS.register_module()
# class ISICDataset(BaseSegDataset):

#     METAINFO = dict(
#         classes=('normal', 'skin lesion'),
#         palette=[[0, 0, 0], [255,255,255]])

#     def __init__(self, img_suffix='.jpg',
#                  seg_map_suffix='_segmentation.png',
#                  reduce_zero_label=False,
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix,
#             seg_map_suffix=seg_map_suffix, 
#             reduce_zero_label=reduce_zero_label,
#             **kwargs)
#         assert fileio.exists(
#             self.data_prefix['img_path'], backend_args=self.backend_args)

@DATASETS.register_module()
class ISICDataset(BaseSegDataset):
    """ISIC2017Task1 dataset.

    In segmentation map annotation for ISIC2017Task1,
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
    """
    METAINFO = dict(classes=('normal', 'skin lesion'),
                    # palette=[[0, 0, 0], [255, 255, 255]])
                    palette=[[120, 120, 120], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
