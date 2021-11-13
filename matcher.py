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
        targets_mentions = targets['mentions']
        bs = outputs["coref_logits"].shape[0]
        matched_predicted_cluster_id = []
        matched_gold_cluster_id = []
        for i in range(bs):
            if targets_clusters[i].shape[1] == 0 or sum(sum(targets_clusters[i])) == 0:
                matched_predicted_cluster_id.append(False)
                matched_gold_cluster_id.append(False)
                continue

            coref_logits = outputs["coref_logits"][i]
            if coref_logits.shape[0] > 1:
                coref_logits = coref_logits.squeeze(0)  # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i].squeeze() # [num_queries]
            if sum(cluster_logits.shape) == 0:
                cluster_logits = torch.tensor([cluster_logits])
            if self.args.add_junk:
                mention_logits = outputs["mention_logits"][i].squeeze(-1).unsqueeze(0) # [1, tokens]

            if not self.args.use_gold_mentions:  #TODO: implement
                # real_cluster_target_rows = torch.sum(targets, -1) > 0
                # real_cluster_target = targets[real_cluster_target_rows]
                # num_of_gold_clusters = int(real_cluster_target.shape[0])
                # num_queries, doc_len = coref_logits.shape

                # cost_coref = []
                # for cluster in real_cluster_target:
                #     gold_per_token_repeated = cluster.repeat(num_queries, 1) # [num_queries, tokens]
                #     losses_for_current_gold_cluster = F.binary_cross_entropy(coref_logits, gold_per_token_repeated, reduction='none').sum(1)

                #     cost_coref.append(losses_for_current_gold_cluster) # [num_queries]
                # cost_coref = torch.stack(cost_coref, 1) # [num_queries, gold_clusters]
            
                # total_cost = self.cost_coref * cost_coref
                pass
            else:
                real_cluster_target_rows = torch.sum(targets_clusters[i], -1) > 0
                real_cluster_target = targets_clusters[i][real_cluster_target_rows]
                num_of_gold_clusters = int(real_cluster_target.shape[0])
                num_queries, doc_len = coref_logits.shape

                cost_is_cluster = F.binary_cross_entropy(cluster_logits, torch.ones_like(cluster_logits), reduction='none') # [num_queries, 1]
                cost_is_cluster = cost_is_cluster.repeat(1, num_of_gold_clusters) # [num_queries, gold_clusters]

                if self.args.add_junk:
                    mention_logits = mention_logits.repeat(num_queries, 1) # [num_queries, tokens]
                    coref_logits = coref_logits * mention_logits

                coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, real_cluster_target.shape[1]).to(coref_logits.device))

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
                    losses_for_current_gold_cluster = F.binary_cross_entropy(coref_logits, gold_per_token_repeated, reduction='none').mean(1)

                    cost_coref.append(losses_for_current_gold_cluster) # [num_queries]
                cost_coref = torch.stack(cost_coref, 1) # [num_queries, gold_clusters]

                total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
                # total_cost = self.cost_coref * cost_coref
            
            total_cost = total_cost.cpu()
            indices = linear_sum_assignment(total_cost)
            ind1, ind2 = indices

            matched_predicted_cluster_id.append(torch.as_tensor(ind1, dtype=torch.int64))
            matched_gold_cluster_id.append(torch.as_tensor(ind2, dtype=torch.int64))

        return matched_predicted_cluster_id, matched_gold_cluster_id


class OrderedMatcher(nn.Module):
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
        matched_predicted_cluster_id = []
        matched_gold_cluster_id = []
        bs = outputs["coref_logits"].shape[0]
        targets_clusters = targets['clusters']

        for i in range(bs):
            matched_predicted_cluster_id.append(torch.arange(0, sum(torch.sum(targets_clusters[i], -1) > 0)))
            matched_gold_cluster_id.append(torch.arange(0, sum(torch.sum(targets_clusters[i], -1) > 0)))

        return matched_predicted_cluster_id, matched_gold_cluster_id


def build_matcher(args, type='Hungarian'):
    if type == 'Hungarian':
        return HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
    else:
        return OrderedMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
