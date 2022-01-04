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

    def __init__(self, cost_is_cluster: float = 1, cost_coref: float = 1, cost_is_mention: float = 1, args=None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_is_cluster = cost_is_cluster
        self.cost_coref = cost_coref
        self.cost_is_mention = cost_is_mention
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
        matched_predicted_cluster_id_real, matched_gold_cluster_id_real, matched_predicted_cluster_id_junk, matched_gold_cluster_id_junk = [],[],[],[]
        for i in range(bs):
            if targets_clusters[i].shape[1] == 0 or torch.sum(targets_clusters[i]) == 0:
                matched_predicted_cluster_id_real.append(False)
                matched_gold_cluster_id_real.append(False)
                matched_predicted_cluster_id_junk.append(False)
                matched_gold_cluster_id_junk.append(False)
                continue

            coref_logits = outputs["coref_logits"][i].squeeze(0) # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i] # [num_queries, 1]
            # if len(outputs["mention_logits"]) > 0:
            #     mention_logits = outputs["mention_logits"][i].squeeze(-1).unsqueeze(0) # [1, tokens]

            if self.args.BIO == 1:
                real_cluster_target_rows = torch.sum(targets_clusters[i], -1) > 0
                real_token_target_cols = torch.sum(targets_clusters[i], -2) > 0
            else:
                real_cluster_target_rows = torch.sum(targets_clusters[i]==2, 1) != targets_clusters[i].shape[1]
                real_token_target_cols = torch.sum(targets_clusters[i]==2, 0) != targets_clusters[i].shape[0]
            real_cluster_target = targets_clusters[i][real_cluster_target_rows]
            num_of_gold_clusters = int(real_cluster_target.shape[0])
            num_queries = coref_logits.shape[0]

            if self.args.is_cluster and (not self.args.use_gold_mentions or self.args.add_junk):
                weight_cluster = torch.cat([torch.ones(num_of_gold_clusters, device=cluster_logits.device), \
                    self.args.eos_coef * torch.ones(num_queries - num_of_gold_clusters, device=cluster_logits.device)]).repeat(num_queries, 1)
                cost_is_cluster = F.binary_cross_entropy(cluster_logits.repeat(1, num_queries), \
                    torch.cat([torch.ones([num_queries, num_of_gold_clusters], device=coref_logits.device), \
                        torch.zeros([num_queries, num_queries - num_of_gold_clusters], device=coref_logits.device)], 1), weight=weight_cluster, reduction='none') # [num_queries, num_queries]
            else:
                cost_is_cluster = torch.tensor(0)

            # if len(outputs["mention_logits"]) > 0:
            #     mention_logits = mention_logits.repeat(num_queries, 1) # [num_queries, tokens]
            #     coref_logits = coref_logits * mention_logits                

            # coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, real_cluster_target.shape[1]).to(coref_logits.device))


        #TODO - bce - sum/mean/dim?
        #TODO: loss for i not after b?
            cost_coref = []
            for cluster in real_cluster_target:
                gold_per_token_repeated = cluster.repeat(num_queries, 1) # [num_queries, tokens]
                if self.args.BIO == 3:
                    losses_for_current_gold_cluster = F.cross_entropy(coref_logits.reshape([-1, 3]), gold_per_token_repeated.reshape([-1]), reduction='none').reshape([coref_logits.shape[0], -1]).mean(1)
                elif self.args.cluster_block:
                    losses_for_current_gold_cluster = F.binary_cross_entropy(cluster_logits * coref_logits, gold_per_token_repeated, reduction='none').mean(1)
                else:
                    losses_for_current_gold_cluster = F.binary_cross_entropy(coref_logits, gold_per_token_repeated, reduction='none').mean(1)
                cost_coref.append(losses_for_current_gold_cluster) # [num_queries]
            if num_of_gold_clusters < num_queries:
                if self.args.BIO == 3: 
                    # zero_cluster = torch.ones_like(targets_clusters[i]) * 2
                    # junk_cluster_score = F.cross_entropy(coref_logits.reshape([-1, 3]), zero_cluster.reshape([-1]), reduction='none').reshape([coref_logits.shape[0], -1]).mean(-1)
                    pass
                else:
                    if self.args.cluster_block:
                        zero_cluster = torch.zeros_like(targets_clusters[i])
                        junk_cluster_score = F.binary_cross_entropy(cluster_logits * coref_logits, zero_cluster, reduction='none').mean(-1)    
                    else:
                        if self.args.slots:
                            zero_cluster = torch.zeros_like(targets_clusters[i].transpose(0,1)[real_token_target_cols].transpose(0,1))
                            junk_cluster_score = F.binary_cross_entropy(coref_logits.transpose(0,1)[real_token_target_cols].transpose(0,1), zero_cluster, reduction='none').mean(-1)
                        else:
                            zero_cluster = torch.zeros_like(targets_clusters[i])
                            junk_cluster_score = F.binary_cross_entropy(coref_logits, zero_cluster, reduction='none').mean(-1) 
                cost_coref += (num_queries-num_of_gold_clusters) * [junk_cluster_score]
            cost_coref = torch.stack(cost_coref, 1) # [num_queries, num_queries/num_of_gold_clusters]

            total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
            
            total_cost = total_cost.cpu().transpose(0,1)  # [gold, predict]
            indices = linear_sum_assignment(total_cost)
            ind1, ind2 = indices

            matched_gold_cluster_id_real.append(torch.as_tensor(ind1[real_cluster_target_rows.cpu()], dtype=torch.int64, device=coref_logits.device))
            matched_predicted_cluster_id_real.append(torch.as_tensor(ind2[real_cluster_target_rows.cpu()], dtype=torch.int64, device=coref_logits.device))
            if len(ind1) > num_of_gold_clusters:
                matched_gold_cluster_id_junk.append(torch.as_tensor(ind1[~real_cluster_target_rows.cpu()], dtype=torch.int64, device=coref_logits.device))
                matched_predicted_cluster_id_junk.append(torch.as_tensor(ind2[~real_cluster_target_rows.cpu()], dtype=torch.int64, device=coref_logits.device))
            else:
                matched_gold_cluster_id_junk.append(torch.as_tensor(list(range(num_of_gold_clusters, num_queries)), dtype=torch.int64, device=coref_logits.device))
                matched_predicted_cluster_id_junk.append(torch.as_tensor([j for j in range(num_queries) if j not in ind1], dtype=torch.int64, device=coref_logits.device))

        return matched_predicted_cluster_id_real, matched_gold_cluster_id_real, matched_predicted_cluster_id_junk, matched_gold_cluster_id_junk

