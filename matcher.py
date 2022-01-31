# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

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
                 "coref_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "cluster_logits": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

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
        targets_clusters = targets['clusters']
        cluster_logits = outputs["cluster_logits"] # [bs, num_queries, 1]
        coref_logits = outputs["predict_matrix"]  # [bs, num_queries, num_mentions]
        bs = outputs["predict_matrix"].shape[0]

        hasgold_coref_logits = coref_logits.clone()
        hasgold_targets_clusters = targets_clusters.clone()
        hasgold_cluster_logits = cluster_logits.clone()

        matched_predicted_cluster_id = torch.arange(0, coref_logits.shape[1], device=coref_logits.device).\
            unsqueeze(0).repeat(coref_logits.shape[0], 1)
        matched_target_cluster_id = torch.arange(0, targets_clusters.shape[1], device=coref_logits.device).\
            unsqueeze(0).repeat(targets_clusters.shape[0], 1)
        indices_with_clusters = torch.arange(0, bs, device=targets_clusters.device)\
            [torch.sum(targets_clusters, [1,2]) > 0]
        if len(indices_with_clusters) == 0:
            return matched_predicted_cluster_id, matched_target_cluster_id

        hasgold_coref_logits = hasgold_coref_logits[:, :self.args.num_queries, :]
        hasgold_cluster_logits = hasgold_cluster_logits[:, :self.args.num_queries, :]
        if len(indices_with_clusters) < bs:
            hasgold_targets_clusters = hasgold_targets_clusters[indices_with_clusters]
            hasgold_coref_logits = hasgold_coref_logits[indices_with_clusters, :self.args.num_queries, :]
            hasgold_cluster_logits = hasgold_cluster_logits[indices_with_clusters, :self.args.num_queries, :]
        num_gold_mentions = torch.sum(hasgold_targets_clusters,[-1,-2])
        num_gold_clusters = torch.sum(torch.sum(hasgold_targets_clusters, -1) > 0, -1)

        if self.args.use_gold_mentions or self.args.use_topk_mentions:
            cost_is_cluster = F.binary_cross_entropy(hasgold_cluster_logits, torch.ones_like(hasgold_cluster_logits), reduction='none') # [bs, num_queries, 1]
            cost_is_cluster = cost_is_cluster.repeat(1, 1, hasgold_targets_clusters.shape[1]) # [bs, num_queries, gold_clusters]            

            cluster_repeated = hasgold_targets_clusters.unsqueeze(1).repeat(1, hasgold_coref_logits.shape[1], 1, 1)[:, :, :, :-1]
            coref_logits_repeated = hasgold_coref_logits.unsqueeze(2).repeat(1, 1, hasgold_targets_clusters.shape[1], 1)

            coref_bce_denom = torch.maximum(num_gold_mentions, torch.ones_like(num_gold_mentions))
            coref_bce_denom = coref_bce_denom.unsqueeze(-1).repeat(1,coref_logits_repeated.shape[1]).unsqueeze(-1)

            cluster_logits_repeated = hasgold_cluster_logits.repeat(1, 1, coref_logits_repeated.shape[2]).unsqueeze(-1)
            clamped_logits = (cluster_logits_repeated * coref_logits_repeated[:, :, :, :-1]).clamp(max=1.0)
            cost_coref = torch.sum(F.binary_cross_entropy(clamped_logits, cluster_repeated, reduction='none'),-1) / \
                coref_bce_denom
            cost_coref += cluster_logits_repeated.squeeze(-1) * coref_logits_repeated[:, :, :, -1]

            total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
            total_cost = total_cost.cpu()
            for j, b in enumerate(indices_with_clusters):
                cur_total_cost = total_cost[j, :, :num_gold_clusters[j]]
                indices = linear_sum_assignment(cur_total_cost)
                ind1, ind2 = indices

                full_ind1 = [*ind1,*[i for i in range(matched_predicted_cluster_id.shape[1]) if i not in ind1]]
                full_ind2 = [*ind2,*[i for i in range(matched_target_cluster_id.shape[1]) if i not in ind2]]

                matched_predicted_cluster_id[b] = torch.tensor(full_ind1, device=matched_predicted_cluster_id.device)
                matched_target_cluster_id[b] = torch.tensor(full_ind2, device=matched_target_cluster_id.device)
                # coref_logits[b, :len(full_ind1)] = coref_logits[b, full_ind1, :]
                # cluster_logits[b, :len(full_ind1)] = cluster_logits[b, full_ind1, :]
                # targets_clusters[b] = targets_clusters[b, full_ind2, :]

        return matched_predicted_cluster_id, matched_target_cluster_id


def build_matcher(args):
    return HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
