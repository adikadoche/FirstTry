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
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Linear(hidden_size, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.is_cluster = nn.Linear(hidden_dim, 1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args
        if args.single_distribution_queries:
            self.query_mu = nn.Parameter(torch.randn(1, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(1, hidden_dim))
        else:
            self.query_mu = nn.Parameter(torch.randn(num_queries, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.word_attn_projection = nn.Linear(hidden_size, 1)
        self.span_width_embed = nn.Embedding(30, 20)
        self.span_proj = nn.Linear(3*hidden_size+20, hidden_dim) # TODO config
             
        self.mention_classifier = nn.Linear(hidden_dim, 1)

        self.IO_score = nn.Sequential(
            nn.Linear(2*hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 1),   #TODO: change to 3 so it would be BIO instead of IO
        ) #query and token concatenated, resulting in IO score

        self.query_head = nn.Linear(hidden_dim, 75)
        self.token_head = nn.Linear(hidden_dim, 75)
        self.query_token_IO_score = nn.Linear(150, 1)  #TODO: change to 3 so it would be BIO instead of IO

        if self.args.slots:
            dim = args.hidden_dim
            self.num_slots = self.num_queries
            self.iters = 3
            self.eps = 1e-8
            self.scale = dim ** -0.5

            self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
            init.xavier_uniform_(self.slots_logsigma)

            self.to_q = nn.Linear(dim, dim)
            self.to_k = nn.Linear(dim, dim)
            self.to_v = nn.Linear(dim, dim)

            self.gru = nn.GRUCell(dim, dim)

            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(dim * 2, dim)
            )

            self.norm_input = nn.LayerNorm(dim)
            self.norm_slots = nn.LayerNorm(dim)
            self.norm_pre_ff = nn.LayerNorm(dim)

            self.mlp_classifier = nn.Sequential(
                nn.Linear(dim, int(dim / 2)),
                nn.ReLU(inplace=True),
                nn.Linear(int(dim / 2), 1),
                nn.Sigmoid()
            ) 

    def forward(self, input_ids, max_mentions_len, mask, gold_mentions, gold_mentions_mask, num_mentions):
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
        span_starts, span_ends, span_mask, longfomer_output, cost_is_mention = self.backbone(input_ids, mask, gold_mentions, gold_mentions_mask)
        new_num_mentions = torch.sum(span_mask, -1, dtype=torch.long)
        span_emb = self.get_span_emb(longfomer_output, span_starts, span_ends)  # [mentions, emb']
        span_emb_proj = self.span_proj(span_emb) # [mentions, emb]
        # mentions = [torch.cat([span_starts[i].unsqueeze(-1), span_ends[i].unsqueeze(-1)], -1) for i in range(bs)]
        cluster_logits, coref_logits, mention_logits = self.slot_attention(span_emb_proj, max_mentions_len[0])
        mentions = torch.cat([torch.cat([span_starts.unsqueeze(-1), span_ends.unsqueeze(-1)], -1), \
            torch.zeros(bs, max_mentions_len[0] - new_num_mentions[0], 2, device=span_starts.device, dtype=torch.long)], 1)
        span_mask = torch.cat([span_mask, torch.zeros(bs, max_mentions_len[0] - span_mask.shape[-1], device=span_mask.device, dtype=torch.long)], -1)

        out = {"coref_logits": coref_logits,
                "cluster_logits": cluster_logits,
                "mention_logits": mention_logits, 
                'mentions': mentions, 
                "span_mask": span_mask}

        if self.args.use_topk_mentions:
            out.update({'cost_is_mention': cost_is_mention})
        return out

    def slot_attention(self, input_emb, max_mentions):
        bs, doc_len, emb, device = *input_emb.shape, input_emb.device

        if self.args.random_queries:
            mu = self.slots_mu.expand(bs, self.num_slots, -1)
            sigma = self.slots_logsigma.exp().expand(bs, self.num_slots, -1)

            slots = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            slots = self.query_embed.weight.unsqueeze(0)

        inputs = self.norm_input(input_emb)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, emb),
                slots_prev.reshape(-1, emb)
            )

            slots = slots.reshape(bs, -1, emb)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        coref_logits = dots.softmax(dim=1) + self.eps
        cluster_logits = self.mlp_classifier(slots)

        coref_logits = torch.cat([coref_logits, torch.zeros(1, coref_logits.shape[1], max_mentions-coref_logits.shape[2]).to(coref_logits.device)], dim=2)

        return cluster_logits, coref_logits, torch.tensor([], device=coref_logits.device)

    def calc_cluster_and_coref_logits(self, last_hs, memory, is_gold_mention, span_mask, max_num_mentions):
        # last_hs [bs, num_queries, emb]
        # memory [bs, tokens, emb]

        cluster_logits = self.is_cluster(last_hs).sigmoid()  # [bs, num_queries, 1]
        if self.args.add_junk:
            mention_logits = self.mention_classifier(memory).sigmoid()  # [bs, tokens, 1]

        #TODO: check cross attention? (without values)

        bs = last_hs.shape[0]
        mention_logits_masked = []
        coref_logits = []
        for i in range(bs):
            cur_memory = memory[i][span_mask[i]==1].unsqueeze(0)
            cur_last_hs = last_hs[i].unsqueeze(0)
            num_tokens_or_mentions = cur_memory.shape[1]
            last_hs_tiled = cur_last_hs.unsqueeze(2).repeat(1, 1, num_tokens_or_mentions, 1) # [bs, num_queries, tokens/mentions, emb]
            memory_tiled = cur_memory.unsqueeze(1).repeat(1, self.num_queries, 1, 1) # [bs, num_queries, tokens/mentions, emb]
            coref_features = torch.cat([last_hs_tiled, memory_tiled], -1) # [bs, num_queries, tokens/mentions, 2 * emb]
            coref_logits_unnorm = self.IO_score(coref_features).squeeze(-1) # [bs, num_queries, tokens/mentions, 1]


        # if not args.add_junk: 
            # cur_coref_logits = coref_logits_unnorm.softmax(dim=1)
        # else:
            cur_coref_logits = coref_logits_unnorm.sigmoid()
            coref_logits.append(torch.cat([cur_coref_logits, (torch.ones(1, cur_coref_logits.shape[1], max_num_mentions-cur_coref_logits.shape[2]) * -1).to(cur_coref_logits.device)], dim=2))

            if self.args.add_junk:
                mention_logits_masked.append(torch.cat([mention_logits[i][span_mask[i]==1].unsqueeze(0), (torch.ones(1, max_num_mentions-cur_coref_logits.shape[2], 1) * -1).to(mention_logits.device)], dim=1))
        # if not is_gold_mention:  #TODO: do I want this?
        #     coref_logits = coref_logits * cluster_logits

        if self.args.add_junk:
            mention_logits_masked = torch.cat(mention_logits_masked)

        return cluster_logits, torch.cat(coref_logits), mention_logits_masked

    def get_span_emb(self, context_outputs_list, span_starts, span_ends):
        span_emb_construct = []
        span_start_emb = context_outputs_list[torch.arange(context_outputs_list.shape[0]).unsqueeze(-1), span_starts] # [bs, k, emb]
        span_emb_construct.append(span_start_emb)

        span_end_emb = context_outputs_list[torch.arange(context_outputs_list.shape[0]).unsqueeze(-1), span_ends] # [bs, k, emb]
        span_emb_construct.append(span_end_emb)

        span_width = (1 + span_ends - span_starts).clamp(max=30)  # [bs, k]
        span_width_index = span_width - 1  # [bs, k]
        span_width_emb = self.span_width_embed.weight[span_width_index]
        span_emb_construct.append(span_width_emb)

        mention_word_score = self.get_masked_mention_word_scores(context_outputs_list, span_starts, span_ends)  # [K, T]
        head_attn_reps = torch.matmul(mention_word_score, context_outputs_list)  # [K, emb]
        span_emb_construct.append(head_attn_reps)
        # span_emb_construct.append((genre[i].unsqueeze(0)/1.0).repeat(num_mentions[i], 1))
        span_emb_cat = torch.cat(span_emb_construct, -1)

        return span_emb_cat  # [bs, k, emb]

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.shape[1]  # T
        num_c = span_starts.shape[1]  # NC

        doc_range = torch.arange(0, num_words, device=span_starts.device).unsqueeze(0).repeat(num_c, 1).unsqueeze(0)  # [1, K, T]
        mention_mask = torch.logical_and(doc_range >= span_starts.unsqueeze(-1),
                                      doc_range <= span_ends.unsqueeze(-1))  # [bs, K, T]

        word_attn = self.word_attn_projection(encoded_doc).squeeze(-1)
        mention_word_attn = F.softmax(mention_mask.to(dtype=torch.float32, device=encoded_doc.device).log() + word_attn.unsqueeze(0), -1)
        return mention_word_attn  # [bs, K, T]

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


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        permuted_coref_logits, permuted_cluster_logits, permuted_targets_clusters = self.matcher(outputs, targets)

        targets_clusters = targets['clusters']
        bs = permuted_coref_logits.shape[0]
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[], 'loss_embedding_num':[], 'loss_embedding_denom':[], 'loss_embedding':[]}

        gold_is_cluster_bool = torch.sum(permuted_targets_clusters, -1) > 0
        gold_is_cluster = torch.zeros_like(permuted_cluster_logits)
        weight_cluster = self.eos_coef * torch.ones_like(permuted_cluster_logits)

        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i].squeeze() # [num_queries]
            num_queries, doc_len = coref_logits.shape
            #TODO: normalize according to number of clusters? (identical to DETR)

            gold_is_cluster = torch.zeros_like(cluster_logits)
            weight_cluster = self.eos_coef * torch.ones_like(cluster_logits)
            if matched_predicted_cluster_id[i] is not False:
                gold_is_cluster[matched_predicted_cluster_id[i]] = 1
                weight_cluster[matched_predicted_cluster_id[i]] = 1
            cost_is_cluster = F.binary_cross_entropy(cluster_logits, gold_is_cluster, weight=weight_cluster)
                
            if self.args.use_topk_mentions and not self.args.is_frozen:
                cost_is_mention = outputs["cost_is_mention"][i]
            else:
                cost_is_mention = torch.tensor(0)

            coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, targets_clusters[i].shape[1]).to(coref_logits.device))

            cost_coref = torch.tensor(0)
            if matched_predicted_cluster_id[i] is not False:
                permuted_coref_logits = coref_logits[matched_predicted_cluster_id[i].numpy()]
                permuted_gold = targets_clusters[i][matched_gold_cluster_id[i].numpy()]
                if self.args.cluster_block:
                    premuted_cluster_logits = cluster_logits[matched_predicted_cluster_id[i].numpy()]
                    cost_coref = F.binary_cross_entropy(premuted_cluster_logits.unsqueeze(1) * permuted_coref_logits,
                                                        permuted_gold, reduction='mean')
                else:
                    cost_coref = F.binary_cross_entropy(permuted_coref_logits, permuted_gold, reduction='mean')
            elif coref_logits.shape[1] > 0:
                cost_coref = F.binary_cross_entropy(coref_logits, torch.zeros_like(coref_logits), reduction='mean')

            # embedding = outputs["embedding"][i].squeeze(0)
            # embedding_loss = torch.tensor(0)

            # num_of_gold_clusters = torch.sum(torch.sum(targets_clusters[i], -1) > 0)
            # cluster_inds = targets_clusters[i][:num_of_gold_clusters]
            # if len(embedding.shape) == 1:
            #     embedding = embedding.unsqueeze(0)
            # junk_cluster = 1 - torch.sum(cluster_inds, 0)
            # if torch.sum(junk_cluster) > 0:
            #     cluster_inds = torch.cat([cluster_inds, junk_cluster.unsqueeze(0)]) #TODO: do we really want it?
            # avg_vector = torch.matmul(cluster_inds, embedding) / torch.sum(cluster_inds, 1).reshape(-1, 1)
            # center_clusters_distances = torch.cdist(avg_vector, avg_vector)
            # diffs = 0
            # for x in range(cluster_inds.shape[0]):
            #     diffs += torch.sum(torch.pow((embedding - avg_vector[x]) * cluster_inds[x].unsqueeze(-1), 2)) / torch.sum(cluster_inds[x])
            # incluster_dist = 3 * diffs/cluster_inds.shape[0]
            # outcluster_dist = torch.sum(center_clusters_distances)/(center_clusters_distances.shape[0]*center_clusters_distances.shape[1])
            # if embedding.shape[0] == 1 or outcluster_dist == 0:
            #     embedding_loss = incluster_dist
            # else:
            #     embedding_loss = incluster_dist + 1 / outcluster_dist  #TODO: change to accurate denom?

            # costs_parts['loss_embedding_num'].append(5 * incluster_dist.detach().cpu())
            # costs_parts['loss_embedding_denom'].append(5 * 1 / outcluster_dist.detach().cpu())                        
            # costs_parts['loss_embedding'].append(5 * embedding_loss.detach().cpu())

            costs_parts['loss_is_cluster'].append(self.cost_is_cluster * cost_is_cluster.detach().cpu())
            costs_parts['loss_is_mention'].append(self.cost_coref * cost_is_mention.detach().cpu())
            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            total_cost = self.cost_coref * cost_coref + self.cost_is_cluster * cost_is_cluster + self.cost_coref * cost_is_mention
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

    def forward(self, input_ids, mask, gold_mentions=None, gold_mentions_mask=None):
        if self.args.use_topk_mentions:
            span_starts, span_ends, mentions_mask, longfomer_no_pad_list, cost_is_mention = self.men_proposal(input_ids, mask, gold_mentions, gold_mentions_mask)
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
        num_queries=args.num_queries,
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

    def forward(self, input_ids, attention_mask=None, gold_mentions=None, gold_mentions_mask=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, dim]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output

        # mention scores
        mention_logits, mention_mask = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, attention_mask)

        mention_probs = mention_logits.sigmoid()
        junk_probs = torch.clone(mention_probs)

        gold_start = torch.index_select(gold_mentions, -1, torch.tensor(0, device=gold_mentions.device)).squeeze(-1)
        gold_end = torch.index_select(gold_mentions, -1, torch.tensor(1, device=gold_mentions.device)).squeeze(-1)
        cost_gold = torch.tensor(0)
        if gold_end.shape[1] > 0:
            gold_probs = torch.clone(mention_probs)[torch.arange(input_ids.shape[0]).unsqueeze(-1).expand(input_ids.shape[0], gold_start.shape[1]),
                                                        gold_start, gold_end].reshape(1,-1)
            cost_gold = torch.sum(F.binary_cross_entropy(gold_probs * gold_mentions_mask, torch.ones_like(gold_probs) * gold_mentions_mask, reduction='none')) / \
                torch.sum(gold_mentions_mask)
            junk_probs[torch.arange(input_ids.shape[0]).unsqueeze(-1).expand(input_ids.shape[0], gold_start.shape[1]),
                                                        gold_start, gold_end] = 0
            junk_probs[torch.arange(input_ids.shape[0]), 0, 0] = mention_probs[torch.arange(input_ids.shape[0]), 0, 0]   #resumig (0,0) probs because it's not a mention for sure (speaker header) and if it's in the mask it will zero in the prev line
        junk_probs = torch.masked_select(junk_probs, mention_mask==1).reshape(1,-1)
 
        cost_is_mention = cost_gold + F.binary_cross_entropy(junk_probs, torch.zeros_like(junk_probs))
       
        return (mention_start_ids, mention_end_ids, span_mask, sequence_output, cost_is_mention.unsqueeze(0))
 