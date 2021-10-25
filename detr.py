# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from consts import OUT_KEYS, TOKENS_PAD
import math


import numpy as np

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
#
# from .backbone import build_backbone
from matcher import build_matcher
from transformer import build_transformer
from transformers import LongformerModel
from typing import List
from transformers import AutoConfig, CONFIG_MAPPING
from misc import NestedTensor, accuracy

logger = logging.getLogger(__name__)


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, args, aux_loss=False):
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
        self.input_proj = nn.Linear(backbone.config.hidden_size, hidden_dim)
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

        self.word_attn_projection = nn.Linear(backbone.config.hidden_size, 1)
        self.span_width_embed = nn.Embedding(30, 20)
        self.span_proj = nn.Linear(3*backbone.config.hidden_size+20, hidden_dim) # TODO config
             
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
 

    def forward(self, input_ids, sum_text_len, mask, gold_mentions):
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
        # input_ids, mask = self.make_batch_same_len(input_ids, mask, sum_text_len)
        input_ids_cat = torch.cat(input_ids)
        mask_cat = torch.cat(mask)
        longformer_emb = self.backbone(input_ids_cat, attention_mask=mask_cat)[0]  # Getting representation for each token in the text

        longformer_emb_size = longformer_emb.shape[-1]
        # filter out masked tokens
        longformer_emb = torch.masked_select(longformer_emb, mask_cat.unsqueeze(-1)==1).reshape(-1, longformer_emb_size)

        if self.args.random_queries:
            raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 1) * self.query_sigma + self.query_mu #raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 0.5)
        else:
            raw_query_embed = self.query_embed.weight

        if not self.args.use_gold_mentions:
            hs, memory = self.transformer(self.input_proj(longformer_emb), mask, raw_query_embed) # [dec_layers, 1, num_queries, emb], [1, seg*seq, emb]
        else:
            span_starts = [torch.tensor([m[0] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_ends = [torch.tensor([m[1] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_emb = self.get_span_emb(longformer_emb, span_starts, span_ends, sum_text_len)  # [mentions, emb']
            span_emb = self.span_proj(span_emb) # [mentions, emb]
            span_emb, span_mask = self.create_mask(span_emb, [len(ss) for ss in span_starts])
            hs, memory = self.transformer(span_emb, span_mask, raw_query_embed)  # [dec_layers, bs, num_queries, emb], [bs, mentions, emb]


        last_hs = hs[-1] # [1, num_queries, emb]
        cluster_logits, coref_logits, mention_logits = self.calc_cluster_and_coref_logits(last_hs, memory, gold_mentions is not None, span_mask)

        # aux_coref_logits = [self.calc_cluster_and_coref_logits(curr_hs, memory)[1] for curr_hs in hs[:-1]]

        # coref_logits = self.temp_embed.weight[:, :doc_len].unsqueeze(0).sigmoid()
        # cluster_logits = self.temp_cluster_embed.weight.unsqueeze(0).sigmoid()
        # coref_logits = coref_logits * cluster_logits

        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        out = {"coref_logits": coref_logits,
                "cluster_logits": cluster_logits,
                "mention_logits": mention_logits}
                # "aux_coref_logits": aux_coref_logits}
        return out

    def create_mask(self, span_emb, mention_num):
        new_span_emb = []
        mask_cat = []
        max_mentions = max(mention_num)
        for i in range(len(mention_num)):
            cur_span_emb = span_emb[i][:mention_num[i]]
            span_mask = torch.cat([torch.ones(cur_span_emb.shape[0], dtype=torch.int, device=span_emb.device), \
                torch.zeros(max_mentions-cur_span_emb.shape[0], dtype=torch.int, device=span_emb.device)])
            cur_span_emb = torch.cat([cur_span_emb, torch.zeros(max_mentions-cur_span_emb.shape[0], cur_span_emb.shape[1], dtype=torch.float, device=span_emb.device)])
            mask_cat.append(span_mask.unsqueeze(0))
            new_span_emb.append(cur_span_emb.unsqueeze(0))
        return torch.cat(new_span_emb), torch.cat(mask_cat)

    def make_batch_same_len(self, input_ids, mask, sum_text_len):
        input_ids_pads = torch.ones(1, self.args.max_segment_len, dtype=torch.int, device=input_ids[0].device) * TOKENS_PAD
        mask_pads = torch.zeros(1, self.args.max_segment_len, dtype=torch.int, device=input_ids[0].device)

        max_seq_num = np.argmax(sum_text_len)
        seq_num = input_ids[max_seq_num].shape[0]

        new_input_ids = []
        new_maks = []
        for i in range(len(input_ids)):
            if input_ids[i].shape[0] < seq_num:
                input_ids[i] = torch.cat([input_ids[i], input_ids_pads.detach().clone().repeat([seq_num - input_ids[i].shape[0], 1])])
                mask[i] = torch.cat([mask[i], mask_pads.detach().clone().repeat([seq_num - mask[i].shape[0], 1])])
            new_input_ids.append(input_ids[i].reshape([1, seq_num*self.args.max_segment_len]))
            new_maks.append(mask[i].reshape([1, seq_num*self.args.max_segment_len]))
        input_ids = torch.cat(new_input_ids)
        mask = torch.cat(new_maks)
        return input_ids, mask

    def calc_cluster_and_coref_logits(self, last_hs, memory, is_gold_mention, span_mask):
        # last_hs [bs, num_queries, emb]
        # memory [bs, tokens, emb]

        cluster_logits = self.is_cluster(last_hs).sigmoid()  # [bs, num_queries, 1]
        mention_logits = self.mention_classifier(memory).sigmoid()  # [bs, tokens, 1]

        #TODO: check cross attention? (without values)

        # if self.args.fc_coref_head:
        #     num_tokens_or_mentions = memory.shape[1]
        #     last_hs_tiled = last_hs.unsqueeze(2).repeat(1, 1, num_tokens_or_mentions, 1) # [bs, num_queries, tokens/mentions, emb]
        #     last_hs_tiled = self.query_head(last_hs_tiled) # [bs, num_queries, tokens/mentions, 75]
        #     memory_tiled = memory.unsqueeze(1).repeat(1, self.num_queries, 1, 1) # [bs, num_queries, tokens/mentions, emb]
        #     memory_tiled = self.token_head(memory_tiled) # [bs, num_queries, tokens/mentions, 75]
        #     coref_features = torch.cat([last_hs_tiled, memory_tiled], -1) # [bs, num_queries, tokens/mentions, 150]
        #     coref_logits_unnorm = self.query_token_IO_score(coref_features).squeeze(-1) # [bs, num_queries, tokens/mentions, 1]
        # else:
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
            coref_logits.append(cur_coref_logits)

            mention_logits_masked.append(mention_logits[i][span_mask[i]==1])

        # if not is_gold_mention:  #TODO: do I want this?
        #     coref_logits = coref_logits * cluster_logits

        return cluster_logits, coref_logits, mention_logits_masked

    def get_span_emb(self, context_outputs, span_starts, span_ends, sum_text_len):
        max_mentions = max([len(s) for s in span_starts])
        # span_mask_list = []
        span_emb_list = []
        end_ind = 0
        for i in range(len(sum_text_len)):
            start_ind = end_ind
            end_ind += sum_text_len[i]
            span_emb_construct = []
            span_start_emb = context_outputs[start_ind + span_starts[i]] # [k, emb]
            span_emb_construct.append(span_start_emb)

            span_end_emb = context_outputs[start_ind + span_ends[i]]  # [k, emb]
            span_emb_construct.append(span_end_emb)

            span_width = (1 + span_ends[i] - span_starts[i]).clamp(max=30)  # [k]

            # if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = self.span_width_embed.weight[span_width_index]
            # TODO add dropout
            # span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_construct.append(span_width_emb)

            # if self.config["model_heads"]:
            mention_word_score = self.get_masked_mention_word_scores(context_outputs[start_ind:end_ind], span_starts[i], span_ends[i])  # [K, T]
            head_attn_reps = torch.matmul(mention_word_score, context_outputs[start_ind:end_ind])  # [K, emb]
            span_emb_construct.append(head_attn_reps)
            span_emb_cat = torch.cat(span_emb_construct, 1)
            # span_mask = torch.cat([torch.ones(span_emb_cat.shape[0], dtype=torch.int, device=context_outputs.device), \
            #     torch.zeros(max_mentions-span_emb_cat.shape[0], dtype=torch.int, device=context_outputs.device)])
            span_emb_cat = torch.cat([span_emb_cat, torch.zeros(max_mentions-span_emb_cat.shape[0], span_emb_cat.shape[1], dtype=torch.float, device=context_outputs.device)])

            span_emb_list.append(span_emb_cat.unsqueeze(0))  
            # span_mask_list.append(span_mask.unsqueeze(0))  
        span_emb_tensor = torch.cat(span_emb_list, 0)
        # span_mask_tensor = torch.cat(span_mask_list, 0)
        return span_emb_tensor#, span_mask_tensor  # [k, emb], [K, T]

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.shape[0]  # T
        num_c = len(span_starts)  # NC

        doc_range = torch.arange(0, num_words).unsqueeze(0).repeat(num_c, 1)  # [K, T]
        mention_mask = torch.logical_and(doc_range >= span_starts.unsqueeze(1),
                                      doc_range <= span_ends.unsqueeze(1))  # [K, T]

        word_attn = self.word_attn_projection(encoded_doc).squeeze(1)
        mention_word_attn = F.softmax(mention_mask.to(dtype=torch.float32, device=encoded_doc.device).log() + word_attn.unsqueeze(0), -1)
        return mention_word_attn  # [K, T]


    def find_mentions(self, hs, memory):
        # cluster memory according to hs. must pass some threshold in order to open cluster and be part of cluster.
        # return list of clusters and the mentions in it
        # pass  #TODO
        # memory size: [text_length, hidden_size]
        # hs size: [n_queries, hidden_size]
        output_logits = hs * memory.transpose() # logits_size = [n_queries, text_length]
        outputs_clusters = F.softmax(output_logits, dim=-1) # output_cluster_size = [text_length, 1] - query assign for each word #TODO decide penalty for non mentions
        return output_logits, outputs_clusters

    @torch.jit.unused
    def _set_aux_loss(self, output_logits, outputs_clusters, outputs_is_cluster):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{OUT_KEYS[0]: a, OUT_KEYS[1]: b, OUT_KEYS[2]: c}
                for a, b, c in zip(output_logits[:-1], outputs_clusters[:-1], outputs_is_cluster[:-1])]


# class SetCriterion(nn.Module):
#     """ This class computes the loss for DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#     def __init__(self, matcher, eos_coef, losses):
#         """ Create the criterion.
#         Parameters:
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             eos_coef: relative classification weight applied to the no-object category
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#         """
#         super().__init__()
#         self.matcher = matcher
#         self.eos_coef = eos_coef
#         self.losses = losses


#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert OUT_KEYS[0] in outputs
#         src_logits = outputs[OUT_KEYS[0]]

#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o

#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         losses = {'loss_ce': loss_ce}

#         if log:
#             # TODO this should probably be a separate loss, not hacked in this one here
#             losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
#         return losses

#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices, num_boxes):
#         """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
#         This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
#         """
#         pred_is_cluster = outputs[OUT_KEYS[2]]
#         device = pred_is_cluster.device
#         tgt_lengths = torch.as_tensor([len(v["is_cluster"]) for v in targets], device=device)
#         # Count the number of predictions that are NOT "no-object" (which is the last class)
#         card_pred = (pred_is_cluster != 0).sum(1)
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         losses = {'cardinality_error': card_err}
#         return losses


#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'cardinality': self.loss_cardinality,
#             'boxes': self.loss_boxes
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != OUT_KEYS[3]}

#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if OUT_KEYS[3] in outputs:
#             for i, aux_outputs in enumerate(outputs[OUT_KEYS[3]]):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         return losses


# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#     @torch.no_grad()
#     def forward(self, outputs, target_sizes):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         out_logits, out_bbox = outputs[OUT_KEYS[0]], outputs[OUT_KEYS[1]]
#
#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2
#
#         prob = F.softmax(out_logits, -1)
#         scores, labels = prob[..., :-1].max(-1)
#
#         # convert to [x0, y0, x1, y1] format
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_sizes.unbind(1)
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]
#
#         results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
#
#         return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / d_model) #TODO: make sure returns float
    return pos * angle_rates


def build_positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): #TODO: replace magic number with text length?
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: NestedTensor):
        xs = self[0](tensor.tensors, tensor.mask)
        out = xs[0]  # [batch_size, seq_len, dim]
        pos = self[1](out)
        # out: List[NestedTensor] = []
        # pos = []
        # for name, x in xs.items():
        #     out.append(x)
        #     # position encoding
        #     pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


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
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        matched_predicted_cluster_id, matched_gold_cluster_id = self.matcher(outputs, targets)

        targets_clusters = targets['clusters']
        targets_mentions = targets['mentions']
        bs = len(targets_clusters)
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[]}
        for i in range(bs):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i].squeeze() # [num_queries]
            mention_logits = outputs["mention_logits"][i].squeeze() # [tokens]
            num_queries, doc_len = coref_logits.shape
            #TODO: normalize according to number of clusters? (identical to DETR)

            # num_of_gold_clusters = len(targets)
            # num_of_gold_clusters = torch.as_tensor([num_of_gold_clusters], dtype=torch.float, device=coref_logits.device)
            # if is_dist_avail_and_initialized():
            #     torch.distributed.all_reduce(num_of_gold_clusters)
            # num_of_gold_clusters = torch.clamp(num_of_gold_clusters / get_world_size(), min=1).item()

            gold_is_cluster = torch.zeros_like(cluster_logits)
            weight_cluster = self.eos_coef * torch.ones_like(cluster_logits)
            if matched_predicted_cluster_id[i] is not False:
                gold_is_cluster[matched_predicted_cluster_id[i]] = 1
                weight_cluster[matched_predicted_cluster_id[i]] = 1
            cost_is_cluster = F.binary_cross_entropy(cluster_logits, gold_is_cluster, weight=weight_cluster)

            if sum(targets_mentions[i].shape) == 0:
                cost_is_mention = 0
            else:
                if sum(mention_logits.shape) == 0:
                    mention_logits = mention_logits.reshape(1)
                weight_mention = targets_mentions[i] + self.eos_coef * (1 - targets_mentions[i])
                cost_is_mention = F.binary_cross_entropy(mention_logits, targets_mentions[i], weight=weight_mention)

            cost_coref = 0
            if matched_predicted_cluster_id[i] is not False:
                permuted_coref_logits = coref_logits[matched_predicted_cluster_id[i].numpy()]
                permuted_gold = targets_clusters[i][matched_gold_cluster_id[i].numpy()]

                if self.args.multiclass_ce:
                    logits = permuted_coref_logits.transpose(0, 1)  # [mentions, num_queries]
                    gold = permuted_gold.transpose(0, 1).nonzero()[:, 1]  # [mentions]
                    cost_coref = F.cross_entropy(logits, gold, reduction='mean')
                else:
                    if self.args.sum_attn:
                        permuted_coref_logits = permuted_coref_logits.clamp(0, 1)
                    cost_coref = F.binary_cross_entropy(permuted_coref_logits, permuted_gold, reduction='mean')
            elif coref_logits.shape[1] > 0:
                cost_coref = F.binary_cross_entropy(coref_logits, torch.zeros_like(coref_logits), reduction='mean')


            # cost_coref = []
            # for predicted_cluster_id, gold_cluster_id in zip(matched_predicted_cluster_id, matched_gold_cluster_id):
            #     # generate gold label for this cluster for all doc tokens
            #     gold_per_token = torch.zeros(doc_len, device=coref_logits.device)
            #     for start, end in targets[gold_cluster_id]:
            #         gold_per_token[start: end + 1] = 1
            #
            #     loss_fn = F.binary_cross_entropy
            #     # loss_fn = F.mse_loss
            #     loss_for_predicted_gold_match = loss_fn(coref_logits[predicted_cluster_id], gold_per_token,
            #                                             reduction='sum')
            #     cost_coref.append(loss_for_predicted_gold_match)
            #
            # cost_coref = torch.stack(cost_coref).sum() if len(cost_coref) > 0 else 0
            costs_parts['loss_is_cluster'].append(self.cost_is_cluster * cost_is_cluster.detach().cpu())
            costs_parts['loss_is_mention'].append(self.cost_is_mention * cost_is_mention.detach().cpu())
            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            total_cost = self.cost_coref * cost_coref + self.cost_is_cluster * cost_is_cluster + self.cost_is_mention * cost_is_mention
            costs.append(total_cost)
        return torch.stack(costs), costs_parts

        # # Compute all the requested losses
        # losses = {}
        # for loss in self.losses:
        #     losses.update(self.get_loss(loss, outputs, targets, indices, num_of_gold_clusters))
        #
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             if loss == 'masks':
        #                 # Intermediate masks losses are too costly to compute, we ignore them.
        #                 continue
        #             kwargs = {}
        #             if loss == 'labels':
        #                 # Logging is enabled only for the last layer
        #                 kwargs = {'log': False}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_of_gold_clusters, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        #
        # return losses

def build_backbone(args, config):
    # position_embedding = PositionalEncoding(config.hidden_size)
    model = LongformerModel.from_pretrained(args.model_name_or_path,
                                               config=config,
                                               cache_dir=args.cache_dir)
    # model = Joiner(backbone, position_embedding)
    # model.backbone_hidden_size = config.hidden_size
    return model


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

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    backbone = build_backbone(args, config)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
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

    return model, criterion #, postprocessors
