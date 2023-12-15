# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr_cls import DETRCLS


@DETECTORS.register_module()
class DeformableDETRCLS(DETRCLS):

    def __init__(self, *args, **kwargs):
        super(DETRCLS, self).__init__(*args, **kwargs)
