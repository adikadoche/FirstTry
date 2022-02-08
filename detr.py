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
import copy

logger = logging.getLogger(__name__)
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, hidden_size, args, aux_loss=False):
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
        # self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.input_proj = nn.Linear(hidden_size, hidden_dim)
        self.slots_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.is_cluster_score = nn.Linear(hidden_dim, 1) 
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args
        # if args.single_distribution_queries:
        #     self.query_mu = nn.Parameter(torch.randn(1, hidden_dim))
        #     self.query_sigma = nn.Parameter(torch.randn(1, hidden_dim))
        # else:
        #     self.query_mu = nn.Parameter(torch.randn(num_queries, hidden_dim))
        #     self.query_sigma = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.span_word_attn_projection = nn.Linear(hidden_size, 1) #
        self.span_width_embed = nn.Embedding(30, 20) #
        self.span_proj = nn.Linear(3*hidden_size+20, hidden_dim) # TODO config #
        self.span_self_attentions = _get_clones(nn.MultiheadAttention(hidden_dim, 1), 3)
             
        # self.mention_classifier = nn.Linear(hidden_dim, 1)

        self.IO_score = nn.Sequential(   
            nn.Linear(2*hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1),   #TODO: change to 3 so it would be BIO instead of IO
        ) #query and token concatenated, resulting in IO score

        # self.query_head = nn.Linear(hidden_dim, 75)
        # self.token_head = nn.Linear(hidden_dim, 75)
        # self.query_token_IO_score = nn.Linear(150, 1)  #TODO: change to 3 so it would be BIO instead of IO

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

            # self.slots_mlp_classifier = nn.Sequential(
            #     nn.Linear(dim, int(dim / 2)),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(int(dim / 2), 1),
            #     nn.Sigmoid()
            # ) 

    def forward(self, input_ids, max_mentions_len, mask, gold_mentions, num_mentions):
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
        # input_ids_cat = torch.cat(input_ids, dim=1).squeeze(0)
        # mask_cat = torch.cat(mask, dim=1).squeeze(0)
        bs = input_ids.shape[0]
        longfomer_no_pad_list, span_starts, span_ends, mentions, cost_is_mention = [[]]*bs, [[]]*bs, [[]]*bs, [[]]*bs, [[]]*bs
        new_num_mentions = torch.zeros(bs, dtype=torch.long)
        for i in range(bs):
            masked_ids = input_ids[i][mask[i]==1].unsqueeze(0)
            masked_mask = torch.ones_like(masked_ids).unsqueeze(0)
            span_starts[i], span_ends[i], mentions_mask, longfomer_no_pad_list[i], cost_is_mention[i] = self.backbone(masked_ids, masked_mask, gold_mentions[i])
            longfomer_no_pad_list[i] = longfomer_no_pad_list[i].squeeze(0)
            span_starts[i] = span_starts[i].squeeze(0)
            span_ends[i] = span_ends[i].squeeze(0)
            new_num_mentions[i] = torch.sum(mentions_mask)
            # start, end = span_starts[i].detach().cpu().numpy(), span_ends[i].detach().cpu().numpy()
            # mentions[i] = [(start[j], end[j]) for j in range(span_starts[i].shape[0])]
        span_emb, span_mask = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends, new_num_mentions)  # [mentions, emb']
        span_emb_proj = self.span_proj(span_emb) # [mentions, emb]
        for i in range(len(self.span_self_attentions)):
            span_emb_proj = self.span_self_attentions[i](span_emb_proj, span_emb_proj, span_emb_proj)[0]
        if max_mentions_len[0] == -1:
            max_mentions_len *= -1 * new_num_mentions[0]
        # mentions = [torch.cat([span_starts[i].unsqueeze(-1), span_ends[i].unsqueeze(-1)], -1) for i in range(bs)]
        inputs, coref_logits, mention_logits = self.slot_attention(span_emb_proj, max_mentions_len[0])
        mentions = torch.cat([\
            torch.cat([\
                torch.cat([span_starts[i].unsqueeze(-1), span_ends[i].unsqueeze(-1)], -1), \
                    torch.ones(max_mentions_len[0] - new_num_mentions[i], 2, device=span_starts[i].device, dtype=torch.long)*-1], 0).unsqueeze(0)\
                            for i in range(bs)], 0)

        cost_is_mention = torch.cat(cost_is_mention, 0)
        out = {"coref_logits": coref_logits,
                "inputs": inputs,
                "mention_logits": mention_logits, 
                'mentions': mentions,
                'cost_is_mention': cost_is_mention}
        return out

    def slot_attention(self, input_emb, max_mentions):
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

        coref_logits = torch.cat([coref_logits, (torch.ones(1, coref_logits.shape[1], max_mentions-coref_logits.shape[2]) * -1).to(coref_logits.device)], dim=2)
        inputs = torch.cat([inputs, (torch.ones(1, max_mentions-inputs.shape[1], inputs.shape[2]) * -1).to(inputs.device)], dim=1)

        return inputs, coref_logits, torch.tensor([], device=coref_logits.device)

    def get_span_emb(self, context_outputs_list, span_starts, span_ends, num_mentions):
        max_mentions = num_mentions.max()
        span_mask_list = []
        span_emb_list = []
        # print(f'context outputs {context_outputs.shape}')
        # print(f'span_starts {span_starts[0].shape}')
        for i in range(len(num_mentions)):
            span_emb_construct = []
            # print(f'span_starts max {span_starts[i].max()} min {span_starts[i].min()}')
            span_start_emb = context_outputs_list[i][span_starts[i][:num_mentions[i]]] # [k, emb]
            span_emb_construct.append(span_start_emb)

            span_end_emb = context_outputs_list[i][span_ends[i][:num_mentions[i]]]  # [k, emb]
            span_emb_construct.append(span_end_emb)

            span_width = (1 + span_ends[i][:num_mentions[i]] - span_starts[i][:num_mentions[i]]).clamp(max=30)  # [k]

            # if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = self.span_width_embed.weight[span_width_index]
            # TODO add dropout
            # span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_construct.append(span_width_emb)

            # if self.config["model_heads"]:
            mention_word_score = self.get_masked_mention_word_scores(context_outputs_list[i], span_starts[i][:num_mentions[i]], span_ends[i][:num_mentions[i]])  # [K, T]
            head_attn_reps = torch.matmul(mention_word_score, context_outputs_list[i])  # [K, emb]
            span_emb_construct.append(head_attn_reps)
            # span_emb_construct.append((genre[i].unsqueeze(0)/1.0).repeat(num_mentions[i], 1))
            span_emb_cat = torch.cat(span_emb_construct, 1)
            span_mask = torch.cat([torch.ones(span_emb_cat.shape[0], dtype=torch.int, device=context_outputs_list[i].device), \
                torch.zeros(max_mentions-span_emb_cat.shape[0], dtype=torch.int, device=context_outputs_list[i].device)])
            span_emb_cat = torch.cat([span_emb_cat, torch.zeros(max_mentions-span_emb_cat.shape[0], span_emb_cat.shape[1], dtype=torch.float, device=context_outputs_list[i].device)])

            span_emb_list.append(span_emb_cat.unsqueeze(0))  
            span_mask_list.append(span_mask.unsqueeze(0))  
        span_emb_tensor = torch.cat(span_emb_list, 0)
        span_mask_tensor = torch.cat(span_mask_list, 0)
        return span_emb_tensor, span_mask_tensor  # [k, emb], [K, T]

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
        costs_parts = {'loss_is_mention':[], 'loss_coref':[], 'loss_junk':[]}
        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries+num_junk_queries, tokens]
            coref_logits = coref_logits[:, :targets_clusters[i].shape[1]]
            #TODO: normalize according to number of clusters? (identical to DETR)
                
            if self.args.use_topk_mentions and not self.args.is_frozen:
                cost_is_mention = outputs["cost_is_mention"][i]
            else:
                cost_is_mention = torch.tensor(0., device=coref_logits.device)

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
                cost_coref += .5 * torch.sum(dist_matrix[i] * goldgold_dist_mask[i]) / goldgold_denom
                passed_thresh = torch.maximum(torch.zeros_like(dist_matrix[i]), \
                    (.3-dist_matrix[i]) * junkgold_dist_mask[i])
                junkgold_denom = torch.sum(passed_thresh>0)
                junkgold_denom = torch.maximum(torch.ones_like(junkgold_denom), junkgold_denom)
                cost_junk += .5 * torch.sum(passed_thresh) / junkgold_denom  #TODO implement dbscan in predict clusters/slot attention?
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

            costs_parts['loss_is_mention'].append(self.cost_is_mention * cost_is_mention.detach().cpu())
            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            costs_parts['loss_junk'].append(self.cost_coref * cost_junk.detach().cpu())
            total_cost = self.cost_coref * cost_coref + \
                self.cost_is_mention * cost_is_mention + self.cost_coref * cost_junk
            costs.append(total_cost)
        return torch.stack(costs), costs_parts

class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.config_name:
            self.config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            self.config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            self.config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
        
        self.men_proposal = None
        self.longformer = None
        if args.use_topk_mentions:
            if args.topk_pre:
                self.men_proposal = MenPropose(AutoConfig.from_pretrained('allenai/longformer-large-4096', cache_dir=args.cache_dir), args) 
                self.men_proposal.load_state_dict(torch.load('/home/gamir/adiz/tmpCode/s2e-coref/s2e_mention_proposal.pt'))
            else:
                self.men_proposal = MenPropose(self.config, args)
            self.men_proposal.top_lambda = args.topk_lambda

            if args.is_frozen:
                for param in self.men_proposal.parameters():
                    param.requires_grad = False
            if args.sep_long:
                self.longformer = LongformerModel.from_pretrained(args.model_name_or_path,
                                                    config=self.config,
                                                    cache_dir=args.cache_dir)
        else:
            self.longformer = LongformerModel.from_pretrained(args.model_name_or_path,
                                                config=self.config,
                                                cache_dir=args.cache_dir)
        self.hidden_size = self.longformer.config.hidden_size if self.longformer is not None else self.men_proposal.longformer.config.hidden_size

    def forward(self, input_ids, mask, gold_clusters=None):
        if self.args.use_topk_mentions:
            span_starts, span_ends, mentions_mask, longfomer_no_pad_list, cost_is_mention = self.men_proposal(input_ids, mask, gold_clusters)
            if self.args.is_frozen and self.longformer is not None:
                longfomer_no_pad_list = self.longformer(input_ids, attention_mask=mask)[0]
            return span_starts, span_ends, mentions_mask, longfomer_no_pad_list, cost_is_mention
        else: 
            return self.longformer(input_ids, mask)

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

    backbone = Backbone(args)

    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries + args.num_junk_queries,
        hidden_size=backbone.hidden_size,
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

class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp

class MenPropose(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_span_length = 30
        self.top_lambda = 0.2
        self.ffnn_size = 3072
        self.do_mlps = True
        self.normalise_loss = True
        self.args = args

        self.longformer = LongformerModel(config)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, 0.3) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, 0.3) if self.do_mlps else None
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, 0.3) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, 0.3) if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]
        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        # span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        span_mask = torch.ones([1, max_k], device=mention_logits.device)  # [batch_size, max_k]   #TODO batch
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        topk_mention_start_ids = (sorted_topk_1d_indices // seq_length).long()  # [batch_size, max_k]
        topk_mention_end_ids = (sorted_topk_1d_indices % seq_length).long()  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                            topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]
        
        # topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits, mention_mask

    def forward(self, input_ids, attention_mask=None, gold_mentions=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, dim]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output

        # mention scores
        mention_logits, mention_mask = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, _ = self._prune_topk_mentions(mention_logits, attention_mask)

        mention_probs = mention_logits.sigmoid()
        junk_probs = torch.clone(mention_probs)

        gold_start = gold_mentions.transpose(0,1)[0]
        gold_start = gold_start[gold_start>0].unsqueeze(0)
        gold_end = gold_mentions.transpose(0,1)[1]
        gold_end = gold_end[gold_end>0].unsqueeze(0)
        cost_gold = torch.tensor(0)
        if gold_end.shape[1] > 0:
            gold_probs = torch.clone(mention_probs)[torch.arange(input_ids.shape[0]).unsqueeze(-1).expand(input_ids.shape[0], gold_start.shape[1]),
                                                        gold_start, gold_end].reshape(1,-1)
            cost_gold = F.binary_cross_entropy(gold_probs, torch.ones_like(gold_probs))
            junk_probs[torch.arange(input_ids.shape[0]).unsqueeze(-1).expand(input_ids.shape[0], gold_start.shape[1]),
                                                        gold_start, gold_end] = 0
        junk_probs = torch.masked_select(junk_probs, mention_mask==1).reshape(1,-1)
 
        cost_is_mention = cost_gold + F.binary_cross_entropy(junk_probs, torch.zeros_like(junk_probs))
       
        return (mention_start_ids, mention_end_ids, span_mask, sequence_output, cost_is_mention.unsqueeze(0))
 