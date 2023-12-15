# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import matplotlib.pyplot as plt
import cv2
import os
import copy
import numpy as np

from sklearn.decomposition import PCA

def pca_vis(tensor, idx=0):
    t = tensor[idx].cpu().numpy()
    c, h, w = t.shape
    t_ = t.reshape(c, -1).transpose(1, 0)

    pca = PCA(n_components=1)
    res = pca.fit_transform(t_, y=0).reshape(h, w)
    return res


@DETECTORS.register_module()
class SingleStageClsDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 classifier=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageClsDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.classifier_input_dim = classifier.get('input_dim', 2048)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.classifier_input_dim, 512, kernel_size=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 3, kernel_size=1, padding=0, bias=True),
        )
            

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        if self.train_cfg != None and self.train_cfg['stage'] == 'resnet_classify':
            self.classifier(x[-1])
        if self.with_neck:
            x = self.neck(x)
        outs = self.bbox_head(x)
        return outs
    
    def visualize(self, x, save_path="tmp.png"):
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(5.0 / 3, 5.0 / 3)  # dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(pca_vis(x[1]))
        fig.savefig("{}".format(save_path), format='jpg', transparent=True, dpi=300, pad_inches = 0)
        plt.close("all")

    def init_visualize(self):
        self.health = np.zeros((6, 100, 100))
        self.sick = np.zeros((6, 100, 100))
        self.tb = np.zeros((6, 100, 100))
        self.health_cnt = 0
        self.sick_cnt = 0
        self.tb_cnt = 0

    def record_visualize(self, x, y):
        target_size = (100, 100)
        resized = []
        for xx in x:
            t = F.interpolate(xx, size=target_size, mode='bilinear', align_corners=False)
            t = pca_vis(t)
            resized.append(t)
        for xx in y:
            t = F.interpolate(xx, size=target_size, mode='bilinear', align_corners=False)
            t = pca_vis(t)
            resized.append(t)
        resized = np.concatenate(resized, axis=0).reshape(6, 100, 100)
        return resized

    def forward_train(self,
                      img,
                      img_metas,
                      gt_classes,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # super(SingleStageClsDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        if self.train_cfg['stage'] == 'resnet_classify' or self.train_cfg['stage'] == 'resnet_finetune':
            img_cls_scores = self.classifier(x[-1])
            img_cls_loss = F.cross_entropy(img_cls_scores.squeeze(), gt_classes.squeeze(),
                                            weight=torch.Tensor([1., 1., 4.]).cuda())
            return {'img_cls_loss': img_cls_loss}
        if self.with_neck:
            x = self.neck(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        # outs = self.bbox_head(x)
        # loss_inputs = outs + (gt_classes, gt_bboxes, gt_labels, img_metas, self.train_cfg)
        # losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def CAM(self, feats, weight, cls=0, save_prefix=None):
            # x: N C 7 7
        # weight: C 512
        feat = feats[0]
        weight = weight[cls, :].view(-1, 1, 1)
        att = feat * weight
        att = att.sum(dim=0).data.cpu().numpy()  # w' * x
        att = (att - att.min()) / (att.max() - att.min() + 1e-15)
        att = cv2.resize(att, (512, 512))


        print("save_prefix", save_prefix)
        if save_prefix is not None:
            plt.axis('off')
            fig = plt.gcf()
            fig.set_size_inches(5.0 / 3, 5.0 / 3)  # dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(att)
            fig.savefig("{}".format(save_prefix), format='jpg', transparent=True, dpi=300, pad_inches=0)
            plt.close('all')
        return att

    def classify(self, img, cls=0, img_meta=None):
        x = self.extract_feat(img)
        if img_meta is not None:
            import os
            save_prefix = "cam_vis/{}".format(os.path.basename(img_meta[0]['filename'][:-4]+"_vis.jpg"))
            if not os.path.exists("cam_vis"):
                os.makedirs("cam_vis")
        else:
            save_prefix = None
        softmax = nn.Softmax(dim=1)
        if "resnet" not in self.test_cfg['stage']:
            img_cls_scores = softmax(self.bbox_head.classifier(x[1]))
            m = self.bbox_head.classifier[0](x[1])
            for module in self.bbox_head.classifier[1:-2]:
                m = module(m)
            att = self.CAM(m, self.bbox_head.classifier[-1].weight, cls=cls, save_prefix=save_prefix)
        else:
            img_cls_scores = softmax(self.classifier(x[-1]))
            m = self.classifier[0](x[-1])
            for module in self.classifier[1:-2]:
                m = module(m)
            att = self.CAM(m, self.classifier[-1].weight, cls=cls, save_prefix=save_prefix)
        return att, img_cls_scores

    def big_plot(self, *args):
        for i in range(len(args)):
            plt.subplot(1, len(args), i+1)
            plt.imshow(args[i], cmap=plt.cm.jet)
        plt.show()
        mmm=1

    @staticmethod
    def plot_composite(img, att):
        plt.imshow(img.detach().cpu().squeeze().numpy().transpose(1,2,0))
        plt.imshow(att, alpha=0.5, cmap=plt.cm.jet)
        plt.show()

    @torch.enable_grad()
    def pre_layercam(self, img, img_meta):
        softmax = nn.Softmax(dim=1)
        self.backbone.record_grad = True
        x = self.backbone(img)
        x = list(x)
        img_cls_scores = softmax(self.classifier1(x[-1]))
        gt_classes = torch.FloatTensor(1, 3).zero_().cuda()
        gt_classes.requires_grad = True
        if 'h' in os.path.basename(img_meta[0]['filename']):
            gt_classes[0][0] = 1 + gt_classes[0][0]
        elif 's' in os.path.basename(img_meta[0]['filename']):
            gt_classes[0][1] = 1 + gt_classes[0][1]
        elif 'tb' in os.path.basename(img_meta[0]['filename']):
            gt_classes[0][2] = 1 + gt_classes[0][2]
        img_cls_scores = img_cls_scores.squeeze()
        gt_classes = gt_classes.squeeze()
        img_cls_scores.backward(gradient=gt_classes, retain_graph=True)

        att = F.relu((x[0] * F.relu(self.backbone.grad[-1])).sum(dim=1))

        att = (att - att.min()) / (att.max() - att.min())

        att = att.squeeze().detach().cpu().numpy()

        #plt.imshow(img)
        plt.imshow(att, cmap=plt.cm.jet)
        #loss = (gt_classes * img_cls_scores.squeeze(dim=2).squeeze(dim=2)).sum()
        #loss = img_cls_scores.mean()

        #loss.backward(retain_graph=True)
        self.backbone.clean_grad()
        return

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        enable_layercam = False
        enable_cam = False
        enable_pca = False
        if enable_layercam:
            self.pre_layercam(img, img_metas)
        
        feat = self.extract_feat(img)

        if enable_cam:
            img_for_show = (img - img.min()) / (img.max() - img.min())
            if "h" in img_metas[0]['filename']:
                #att_error, img_cls_scores_error = self.classify(error_area, cls=1)
                att, img_cls_scores = self.classify(img, cls=0, img_meta=img_metas)

                #self.plot_composite(img_for_show, att)
            if "s" in img_metas[0]['filename']:
                #att_error, img_cls_scores_error = self.classify(error_area, cls=1)
                att, img_cls_scores = self.classify(img, cls=1, img_meta=img_metas)

                #self.plot_composite(img_for_show, att)
                #att_minus, img_cls_scores_minus = self.classify(img_, cls=1)
                #att_none, img_cls_scores_none = self.classify(torch.zeros(1, 3, 512, 512).cuda(), cls=1)

                #self.big_plot(att, att_minus, att_error, att_none)
            if "t" in os.path.basename(img_metas[0]['filename']):
                print("detect a tb image")
                #att_error, img_cls_scores_error = self.classify(error_area, cls=2)
                att, img_cls_scores = self.classify(img, cls=2, img_meta=img_metas)
                #self.plot_composite(img_for_show, att)
                #att_minus, img_cls_scores_minus = self.classify(img_, cls=2)
                #att_none, img_cls_scores_none = self.classify(torch.zeros(1, 3, 512, 512).cuda(), cls=2)
                #self.big_plot(att, att_minus, att_error, att_none)

        img_cls_scores = self.classifier(feat[-1])
        name = os.path.basename(img_metas[0]['filename'])
        
        if self.with_neck:
            feat = self.neck(feat)
        if enable_pca:
            self.visualize(feat, save_path=\
                           "/media/wyh/yuhuan/zsc/mmdetection/mmdetection/vis/pca_vis/{}".\
                            format(name[:-4]+'_vis.jpg'))
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        return bbox_results, img_cls_scores

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        if self.with_neck:
            feats = [self.neck(i) for i in feats]
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        if self.with_neck:
            x = self.neck(x)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
