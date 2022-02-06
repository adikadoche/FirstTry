# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import logging
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, LongformerModel
from transformers.models.bert.modeling_bert import ACT2FN
from torch import nn, Tensor
from consts import OUT_KEYS, TOKENS_PAD
import math


import numpy as np
from matcher import build_matcher
from transformer import build_transformer
from transformers import AutoConfig, CONFIG_MAPPING
from utils import mask_tensor

logger = logging.getLogger(__name__)


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, num_queries, args, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.longformer = LongformerModel(AutoConfig.from_pretrained('allenai/longformer-large-4096'))
        hidden_size = self.longformer.config.hidden_size
        hidden_dim = transformer.d_model
        self.slots_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.aux_loss = aux_loss
        self.args = args

        self.span_word_attn_projection = nn.Linear(hidden_size, 1) #
        self.span_width_embed = nn.Embedding(30, 20) #
        self.span_proj = nn.Linear(3*hidden_size+20, hidden_dim) # TODO config #
             
        if self.args.slots:
            dim = args.hidden_dim
            self.num_slots = self.num_queries
            self.slots_iters = 3
            self.slots_eps = 1e-8
            self.slots_scale = dim ** -0.5

            self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
            init.xavier_uniform_(self.slots_logsigma)

            self.slots_to_q = nn.Linear(dim, dim)
            self.slots_to_k = nn.Linear(dim, dim)
            self.slots_to_v = nn.Linear(dim, dim)

            self.slots_gru = nn.GRUCell(dim, dim)

            self.slots_mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(dim * 2, dim)
            )

            self.slots_norm_input = nn.LayerNorm(dim)
            self.norm_slots = nn.LayerNorm(dim)
            self.slots_norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, input_ids, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        bs = input_ids.shape[0]
        longfomer_no_pad_list, span_starts, span_ends = [[]]*bs, [[]]*bs, [[]]*bs
        for i in range(bs):
            masked_ids = input_ids[i][mask[i]==1].unsqueeze(0)
            masked_mask = torch.ones_like(masked_ids).unsqueeze(0)
            longfomer_no_pad_list[i] = self.longformer(masked_ids, attention_mask=masked_mask)[0]
            longfomer_no_pad_list[i] = longfomer_no_pad_list[i].squeeze(0)

            mention_mask = torch.ones(longfomer_no_pad_list[i].shape[-2], longfomer_no_pad_list[i].shape[-2], dtype=torch.long)
            mention_mask = mention_mask.triu(diagonal=0)
            mention_mask = mention_mask.tril(diagonal=self.args.max_span_length - 1)
            mention_mask = mention_mask.reshape(-1)
            span_starts[i] = torch.arange(0, longfomer_no_pad_list[i].shape[-2]).reshape(-1, 1).\
                repeat(1, longfomer_no_pad_list[i].shape[-2]).reshape(-1)[mention_mask==1]
            span_ends[i] = torch.arange(0, longfomer_no_pad_list[i].shape[-2]).reshape(1, -1).\
                repeat(longfomer_no_pad_list[i].shape[-2], 1).reshape(-1)[mention_mask==1]
            mentions = torch.cat([span_starts[i].unsqueeze(-1), span_ends[i].unsqueeze(-1)], -1)

        span_emb = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends)  # [mentions, emb']
        span_emb_proj = self.span_proj(span_emb) # [mentions, emb]
        inputs, coref_logits, mention_logits = self.slot_attention(span_emb_proj)

        out = {"coref_logits": coref_logits,
                "inputs": inputs,
                "mention_logits": mention_logits,
                "mentions": mentions}
        return out

    def slot_attention(self, input_emb):
        bs, doc_len, emb, device = *input_emb.shape, input_emb.device

        if self.args.random_queries:
            mu = self.slots_mu.expand(bs, self.num_slots, -1)
            sigma = self.slots_logsigma.exp().expand(bs, self.num_slots, -1)

            slots = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            slots = self.slots_query_embed.weight.unsqueeze(0)

        inputs = self.slots_norm_input(input_emb)
        k, v = self.slots_to_k(inputs), self.slots_to_v(inputs)

        for _ in range(self.slots_iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.slots_to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.slots_scale
            attn = dots.softmax(dim=1) + self.slots_eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.slots_gru(
                updates.reshape(-1, emb),
                slots_prev.reshape(-1, emb)
            )

            slots = slots.reshape(bs, -1, emb)
            slots = slots + self.slots_mlp(self.slots_norm_pre_ff(slots))

        slots = self.norm_slots(slots)
        q = self.slots_to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.slots_scale
        coref_logits = (dots.softmax(dim=1) + self.slots_eps).clamp(max=1.0)

        return inputs, coref_logits, torch.tensor([], device=coref_logits.device)

    def get_span_emb(self, context_outputs_list, span_starts, span_ends):
        span_emb_list = []
        for i in range(len(context_outputs_list)):
            span_emb_construct = []
            span_start_emb = context_outputs_list[i][span_starts[i]] # [k, emb]
            span_emb_construct.append(span_start_emb)

            span_end_emb = context_outputs_list[i][span_ends[i]]  # [k, emb]
            span_emb_construct.append(span_end_emb)

            span_width = (1 + span_ends[i] - span_starts[i]).clamp(max=self.args.max_span_length)  # [k]

            span_width_index = span_width - 1  # [k]
            span_width_emb = self.span_width_embed.weight[span_width_index]
            span_emb_construct.append(span_width_emb)

            mention_word_score = self.get_masked_mention_word_scores(context_outputs_list[i], span_starts[i], span_ends[i])  # [K, T]
            head_attn_reps = torch.matmul(mention_word_score, context_outputs_list[i])  # [K, emb]
            span_emb_construct.append(head_attn_reps)
            span_emb_cat = torch.cat(span_emb_construct, 1)

            span_emb_list.append(span_emb_cat.unsqueeze(0))  
        span_emb_tensor = torch.cat(span_emb_list, 0)
        return span_emb_tensor  # [k, emb], [K, T]

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.shape[0]  # T
        num_c = len(span_starts)  # NC

        doc_range = torch.arange(0, num_words, device=span_starts.device).unsqueeze(0).repeat(num_c, 1)  # [K, T]
        mention_mask = torch.logical_and(doc_range >= span_starts.unsqueeze(1),
                                      doc_range <= span_ends.unsqueeze(1))  # [K, T]

        word_attn = self.span_word_attn_projection(encoded_doc).squeeze(1)
        mention_word_attn = F.softmax(mention_mask.to(dtype=torch.float32, device=encoded_doc.device).log() + word_attn.unsqueeze(0), -1)
        return mention_word_attn  # [K, T]

class MatchingLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, eos_coef, cost_is_cluster, cost_coref, cost_is_mention, args):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.cost_is_cluster = cost_is_cluster
        self.cost_coref = cost_coref
        self.cost_is_mention = cost_is_mention
        self.args = args
        self.eos_coef = eos_coef


    def forward(self, outputs, targets, dist_matrix, goldgold_dist_mask, junkgold_dist_mask):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        matched_predicted_cluster_id, matched_gold_cluster_id = self.matcher(outputs, targets)

        targets_clusters = targets['clusters']
        bs = outputs["coref_logits"].shape[0]
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_coref':[], 'loss_junk':[]}
        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries+num_junk_queries, tokens]
            coref_logits = coref_logits[:, :targets_clusters[i].shape[1]]
                
            cost_coref = torch.tensor(0., device=coref_logits.device)
            cost_junk = torch.tensor(0., device=coref_logits.device)
            if matched_predicted_cluster_id[i] is not False:  #TODO: add zero rows?
                permuted_coref_logits = coref_logits[matched_predicted_cluster_id[i].numpy()]
                junk1_coref_logits = coref_logits[[x for x in range(self.args.num_queries) if x not in matched_predicted_cluster_id[i].numpy()]]
                junk2_coref_logits = coref_logits[[x for x in range(self.args.num_queries, coref_logits.shape[0]) if x not in matched_predicted_cluster_id[i].numpy()]]
                permuted_gold = targets_clusters[i][matched_gold_cluster_id[i].numpy()]
                permuted_gold = permuted_gold[:, :-1]
                junk1_gold = torch.zeros_like(junk1_coref_logits)
                junk2_gold = torch.zeros_like(junk2_coref_logits[:, :-1])

                clamped_logits = (permuted_coref_logits[:, :-1]).clamp(max=1.0)
                cost_coref = F.binary_cross_entropy(clamped_logits, permuted_gold, reduction='mean') + \
                                                        torch.mean(permuted_coref_logits[:, -1])
                clamped_junk1_logits = (junk1_coref_logits).clamp(max=1.0)
                cost_junk = F.binary_cross_entropy(clamped_junk1_logits, junk1_gold, reduction='mean')
                clamped_junk2_logits = (junk2_coref_logits[:, :-1]).clamp(max=1.0)
                cost_junk += F.binary_cross_entropy(clamped_junk2_logits, junk2_gold, reduction='mean')
            elif coref_logits.shape[1] > 0:
                clamped_logits = coref_logits.clamp(max=1.0)
                cost_coref = F.binary_cross_entropy(clamped_logits, torch.zeros_like(coref_logits), reduction='mean')

            dist_matrix[i] = dist_matrix[i].clamp(min=0.0, max=1.0)
            goldgold_denom = torch.sum(goldgold_dist_mask[i])
            goldgold_denom = torch.maximum(torch.ones_like(goldgold_denom), goldgold_denom)
            if self.args.loss =='max':
                cost_coref += torch.sum(dist_matrix[i] * goldgold_dist_mask[i]) / goldgold_denom
                passed_thresh = torch.maximum(torch.zeros_like(dist_matrix[i]), \
                    (.3-dist_matrix[i]) * junkgold_dist_mask[i])
                junkgold_denom = torch.sum(passed_thresh>0)
                junkgold_denom = torch.maximum(torch.ones_like(junkgold_denom), junkgold_denom)
                cost_junk += torch.sum(passed_thresh) / junkgold_denom  #TODO implement dbscan in predict clusters/slot attention?
            elif self.args.loss =='div':
                cost_coref += .5 * torch.sum(dist_matrix[i] * goldgold_dist_mask[i]) / goldgold_denom
                # cost_junk = .5 * torch.sum(dist_matrix[i] * junkgold_dist_mask[i]) / junkgold_denom
                junkgold_denom = torch.sum(junkgold_dist_mask[i])
                junkgold_denom = torch.maximum(torch.ones_like(junkgold_denom), junkgold_denom)
                junk_dists = dist_matrix[i] * junkgold_dist_mask[i]
                cost_junk += .5 * torch.sum(1 / junk_dists[junk_dists > 0]) / junkgold_denom
            elif self.args.loss =='bce':
                log_incluster_dists = dist_matrix[i] * goldgold_dist_mask[i]
                log_outcluster_dists = (1-dist_matrix[i]) * junkgold_dist_mask[i]
                cost_coref += F.binary_cross_entropy(log_incluster_dists, torch.zeros_like(log_incluster_dists), \
                    reduction='sum') / goldgold_denom
                junkgold_denom = torch.sum(junkgold_dist_mask[i])
                junkgold_denom = torch.maximum(torch.ones_like(junkgold_denom), junkgold_denom)
                cost_junk += F.binary_cross_entropy(log_outcluster_dists, torch.zeros_like(log_outcluster_dists), \
                    reduction='sum') / junkgold_denom

            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            costs_parts['loss_junk'].append(self.cost_coref * cost_junk.detach().cpu())
            total_cost = self.cost_coref * cost_coref + self.cost_coref * cost_junk
            costs.append(total_cost)
        return torch.stack(costs), costs_parts

def build_DETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)

    transformer = build_transformer(args)

    model = DETR(
        transformer,
        num_queries=args.num_queries + args.num_junk_queries,
        args=args,
        aux_loss=args.aux_loss
    )

    matcher = build_matcher(args)
    # TODO maybe return consideration of aux loss

    criterion = MatchingLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_is_mention=args.cost_is_mention,
                             cost_coref=args.cost_coref, args=args)

    # if args.loss == 'match':
    #     criterion = MatchingLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
    # elif args.loss == 'bcubed':
    #     criterion = BCubedLoss()

    criterion.to(device)
    # postprocessors = {'bbox': PostProcess()}

    return model, criterion
