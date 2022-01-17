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
        bs = outputs["coref_logits"].shape[0]
        matched_predicted_cluster_id = [[]] * bs
        matched_gold_cluster_id = [[]] * bs
        indices_with_clusters = []

        for i in range(bs):
            if torch.sum(targets_clusters[i]) == 0:
                matched_predicted_cluster_id[i] = False
                matched_gold_cluster_id[i] = False
            else:
                indices_with_clusters.append(i)
        if len(indices_with_clusters) == 0:
            return matched_predicted_cluster_id, matched_gold_cluster_id
        indices_with_clusters = torch.tensor(indices_with_clusters, device=targets_clusters.device)

        num_gold_clusters = torch.sum(torch.sum(targets_clusters, -1) > 0, -1)
        max_gold_clusters = torch.max(num_gold_clusters)
        real_cluster_target = targets_clusters[:,:max_gold_clusters,:]  #[bs, gold_clusters, num_mentions]
        cluster_logits = outputs["cluster_logits"] # [bs, num_queries, 1]
        coref_logits = outputs["coref_logits"]  # [bs, num_queries, num_mentions]
        if len(indices_with_clusters) < bs:
            real_cluster_target = torch.index_select(real_cluster_target, 0, indices_with_clusters)
            cluster_logits = torch.index_select(cluster_logits, 0, indices_with_clusters)
            coref_logits = torch.index_select(coref_logits, 0, indices_with_clusters)

        if self.args.use_gold_mentions or self.args.use_topk_mentions:
            cost_is_cluster = F.binary_cross_entropy(cluster_logits, torch.ones_like(cluster_logits), reduction='none') # [bs, num_queries, 1]
            cost_is_cluster = cost_is_cluster.repeat(1, 1, max_gold_clusters) # [bs, num_queries, gold_clusters]            

            cluster_repeated = real_cluster_target.unsqueeze(1).repeat(1, coref_logits.shape[1], 1, 1)
            coref_logits_repeated = coref_logits.unsqueeze(2).repeat(1, 1, real_cluster_target.shape[1], 1)
            if self.args.cluster_block:
                cluster_logits_repeated = cluster_logits.unsqueeze(1).repeat(1, real_cluster_target.shape[0], 1)
                cost_coref = torch.sum(F.binary_cross_entropy(coref_logits_repeated * cluster_logits_repeated, cluster_repeated, reduction='none'),-1)/torch.sum(outputs['span_mask'], 1)
            else:
                cost_coref = torch.sum(F.binary_cross_entropy(coref_logits_repeated, cluster_repeated, reduction='none'),-1)/torch.sum(outputs['span_mask'], 1).unsqueeze(1).unsqueeze(1)  # [bs, num_queries, gold_clusters]

            total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
            total_cost = total_cost.cpu()   # [bs, num_queries, gold_clusters]

            for b in range(bs):
                cur_total_cost = total_cost[b, :, :num_gold_clusters[b]]
                indices = linear_sum_assignment(cur_total_cost)
                ind1, ind2 = indices

                matched_predicted_cluster_id[b] = torch.as_tensor(ind1, dtype=torch.int64)
                matched_gold_cluster_id[b] = torch.as_tensor(ind2, dtype=torch.int64)

        return matched_predicted_cluster_id, matched_gold_cluster_id


def build_matcher(args):
    return HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
