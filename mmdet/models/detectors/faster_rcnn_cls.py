# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage_class import TwoStageCLSDetector


@DETECTORS.register_module()
class FasterRCNNCLS(TwoStageCLSDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 classifier=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNCLS, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            classifier=classifier,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
