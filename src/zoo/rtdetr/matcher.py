"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"].float()
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            logits_b = outputs["pred_logits"][b].float()
            if self.use_focal_loss:
                out_prob = F.sigmoid(logits_b)[:, tgt_ids].clamp(min=1e-6, max=1.0 - 1e-6)
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-out_prob.log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                out_prob = logits_b.softmax(-1)
                cost_class = -out_prob[:, tgt_ids]

            out_bbox = outputs["pred_boxes"][b].float()
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            cost_mat = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
            cost_mat = torch.nan_to_num(cost_mat, nan=1e5, posinf=1e5, neginf=-1e5).cpu()
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return {'indices': indices}
        