# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_is_cluster: float = 1, cost_coref: float = 1, args=None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_is_cluster = cost_is_cluster
        self.cost_coref = cost_coref
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, targets):
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
        coref_logits = outputs["coref_logits"].squeeze(0) # [num_queries, tokens]
        cluster_logits = outputs["cluster_logits"].squeeze(0) # [num_queries, 1]
        real_cluster_target_rows = torch.sum(targets, -1) > 0
        real_cluster_target = targets[real_cluster_target_rows]
        num_of_gold_clusters = int(targets[real_cluster_target_rows].shape[0])
        num_queries, doc_len = coref_logits.shape

        cost_is_cluster = F.binary_cross_entropy(cluster_logits, torch.ones_like(cluster_logits), reduction='none') # [num_queries, 1]
        cost_is_cluster = cost_is_cluster.repeat(1, num_of_gold_clusters) # [num_queries, gold_clusters]

        cost_coref = []
        for cluster in real_cluster_target:
            gold_per_token_repeated = cluster.repeat(num_queries, 1) # [num_queries, tokens]
            if self.args.multiclass_ce:
                # logits = coref_logits.transpose(0, 1)  # [mentions, num_queries]
                # gold = gold_per_token_repeated.transpose(0, 1).nonzero()[:, 1]  # [mentions]
                # cost_coref = F.cross_entropy(logits, gold, reduction='sum')
                coref_logits = coref_logits.softmax(-2)
            if self.args.sum_attn:
                coref_logits = coref_logits.clamp(0, 1)
            losses_for_current_gold_cluster = F.binary_cross_entropy(coref_logits, gold_per_token_repeated, reduction='none').sum(1)

            cost_coref.append(losses_for_current_gold_cluster) # [num_queries]

        cost_coref = torch.stack(cost_coref, 1) # [num_queries, gold_clusters]

        total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
        total_cost = total_cost.cpu()
        indices = linear_sum_assignment(total_cost)
        i, j = indices
        return (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))



        # bs, num_queries = outputs["pred_logits"].shape[:2]
        #
        # # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
        #
        # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # # but approximate it in 1 - proba[target class].
        # # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = -out_prob[:, tgt_ids]
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        #
        # # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        #
        # # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # C = C.view(bs, num_queries, -1).cpu()
        #
        # sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
