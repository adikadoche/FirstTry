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

            coref_logits = outputs["coref_logits"][i].squeeze(0) # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i] # [num_queries, 1]
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

                if self.args.detr:
                    cost_is_cluster = torch.cat([(1 - cluster_logits).repeat(1, num_of_gold_clusters), cluster_logits.repeat(1, num_queries - num_of_gold_clusters)], 1)
                else:
                    cost_is_cluster = F.binary_cross_entropy(cluster_logits.repeat(1, num_queries), \
                        torch.cat([torch.ones([num_queries, num_of_gold_clusters], device=coref_logits.device), \
                            torch.zeros([num_queries, num_queries - num_of_gold_clusters], device=coref_logits.device)], 1), reduction='none') # [num_queries, num_queries]

                if self.args.add_junk:
                    mention_logits = mention_logits.repeat(num_queries, 1) # [num_queries, tokens]
                    coref_logits = coref_logits * mention_logits

                coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, real_cluster_target.shape[1]).to(coref_logits.device))

                cost_coref = []
                for cluster in real_cluster_target:
                    if self.args.detr:
                        losses_for_current_gold_cluster = torch.cdist(coref_logits, cluster.unsqueeze(0), p=1).squeeze()
                    else:
                        gold_per_token_repeated = cluster.repeat(num_queries, 1) # [num_queries, tokens]
                        if self.args.multiclass_ce:
                            # logits = coref_logits.transpose(0, 1)  # [mentions, num_queries]
                            # gold = gold_per_token_repeated.transpose(0, 1).nonzero()[:, 1]  # [mentions]
                            # cost_coref = F.cross_entropy(logits, gold, reduction='sum')
                            coref_logits = coref_logits.softmax(-2)
                        if self.args.sum_attn:
                            coref_logits = coref_logits.clamp(0, 1)
                        if self.args.cluster_block:
                            losses_for_current_gold_cluster = F.binary_cross_entropy(cluster_logits * coref_logits, gold_per_token_repeated, reduction='none').mean(1)
                        else:
                            losses_for_current_gold_cluster = F.binary_cross_entropy(coref_logits, gold_per_token_repeated, reduction='none').mean(1)
                    cost_coref.append(losses_for_current_gold_cluster) # [num_queries]
                if num_of_gold_clusters < num_queries:
                    zero_cluster = torch.zeros([num_queries, real_cluster_target.shape[1]], device=coref_logits.device)
                    if self.args.detr:
                        junk_cluster_score = torch.cdist(coref_logits, zero_cluster[0].unsqueeze(0), p=1).squeeze()
                    elif self.args.cluster_block:
                        junk_cluster_score = F.binary_cross_entropy(cluster_logits * coref_logits, zero_cluster, reduction='none').mean(1)
                    else:
                        junk_cluster_score = F.binary_cross_entropy(coref_logits, zero_cluster, reduction='none').mean(1)
                    cost_coref += (num_queries-num_of_gold_clusters) * [junk_cluster_score]
                cost_coref = torch.stack(cost_coref, 1) # [num_queries, num_queries]

                if self.args.detr:
                    total_cost = self.cost_is_cluster * cost_is_cluster + (self.cost_coref+2) * cost_coref
                else:
                    total_cost = self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_coref
                # total_cost = self.cost_coref * cost_coref
            
            total_cost = total_cost.cpu()
            indices = linear_sum_assignment(total_cost)
            ind1, ind2 = indices

            matched_predicted_cluster_id.append(torch.as_tensor(ind1, dtype=torch.int64, device=coref_logits.device))
            matched_gold_cluster_id.append(torch.as_tensor(ind2, dtype=torch.int64, device=coref_logits.device))

        return matched_predicted_cluster_id, matched_gold_cluster_id



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
