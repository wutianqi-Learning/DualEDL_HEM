from seg.datasets import BaseSegDataset
from seg.registry import DATASETS


@DATASETS.register_module()
class FUSC2021Dataset(BaseSegDataset):
    """FUSC2021Dataset dataset.

    In segmentation map annotation for FUSC2021Dataset, 0 stands for background
    , which is included in 2 categories. ``reduce_zero_label``
    is fixed to False. The ``img_suffix`` is fixed to '.png' and
    ``seg_map_suffix`` is fixed to '.png'.
    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False..
    """
    METAINFO = dict(classes=('background', 'wound'))

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
