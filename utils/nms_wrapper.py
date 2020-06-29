# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import torch
from torchvision.ops import nms


def nms_detections(pred_boxes, scores, nms_thresh):
    pred_boxes = torch.FloatTensor(pred_boxes)
    scores = torch.FloatTensor(scores)
    keep = nms(pred_boxes, scores, nms_thresh)
    return keep
