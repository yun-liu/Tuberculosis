# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_class import SingleStageClsDetector


@DETECTORS.register_module()
class FCOSCLS(SingleStageClsDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 classifier=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOSCLS, self).__init__(backbone, neck, bbox_head, classifier, train_cfg,
                                   test_cfg, pretrained, init_cfg)
