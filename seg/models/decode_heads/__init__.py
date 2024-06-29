# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .uper_head import UPerHead
from .fcn_head_hem import FCNHeadHEM
from .aspp_head_hem import ASPPHeadHEM
from .decode_head_hem import BaseDecodeHeadHEM
from .fcn_head_edl import FCNHeadEDL
from .decode_head_edl import BaseDecodeHeadEDL
from .fcn_head_withconv_dualhem import FCNHeadConvHEM
from .fcn_head_withconv_dualhem_edl import FCNHeadConvHEMEDL
from .decode_head_dualhem_edl import BaseDecodeHeadHEMEDL
__all__ = [
    'ASPPHead', 'FCNHead', 'ASPPHeadHEM',
    'PSPHead', 'UPerHead', 'FCNHeadHEM',
    'BaseDecodeHeadHEM', 'FCNHeadEDL',
    'BaseDecodeHeadEDL', 'FCNHeadConvHEM',
    'BaseDecodeHeadHEMEDL', 'FCNHeadConvHEMEDL'
]
