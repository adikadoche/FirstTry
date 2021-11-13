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
from consts import GENRES

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
        if self.args.speaker == 'before':
            # self.span_proj = nn.Linear(3*backbone.config.hidden_size+20+self.args.max_num_speakers+len(GENRES)+1, hidden_dim) # TODO config
            self.span_proj = nn.Linear(3*backbone.config.hidden_size+20 + self.args.max_num_speakers, hidden_dim) # TODO config
        elif self.args.speaker == 'after':
            self.span_proj = nn.Linear(3*backbone.config.hidden_size+20, hidden_dim - self.args.max_num_speakers) # TODO config
        elif self.args.speaker == 'text':
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
 

    def forward(self, input_ids, sum_text_len, mask, gold_mentions, num_mentions, speaker_ids, genre, gold_matrix, cluster_number):
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
        input_ids_r = input_ids.reshape(input_ids.shape[0], -1)
        mask_r = mask.reshape(mask.shape[0], -1)
        if self.args.speaker != 'text':
            speaker_ids_r = speaker_ids.reshape(speaker_ids.shape[0], -1, speaker_ids.shape[-1])
        longfomer_no_pad_list = []
        speaker_ids_no_pad_list = []
        for i in range(input_ids_r.shape[0]):
            masked_ids = input_ids_r[i][mask_r[i]==1].unsqueeze(0)
            masked_mask = torch.ones_like(masked_ids).unsqueeze(0)
            if masked_ids.shape[-1] > self.args.max_seq_length:
                masked_ids = torch.zeros([2, math.ceil(input_ids.shape[1]/2) * input_ids.shape[-1]], dtype=torch.long)
                masked_mask = torch.zeros([2, math.ceil(mask.shape[1]/2) * mask.shape[-1]], dtype=torch.long)
                masked_ids[0] = input_ids[i][:math.ceil(input_ids.shape[1]/2)].reshape(1, math.ceil(input_ids.shape[1]/2) * input_ids.shape[-1])
                masked_mask[0] = mask[i][:math.ceil(mask.shape[1]/2)].reshape(1, math.ceil(mask.shape[1]/2) * mask.shape[-1])
                masked_ids[1][:(input_ids.shape[1]-math.ceil(input_ids.shape[1]/2)) * input_ids.shape[-1]] = \
                    input_ids[i][math.ceil(input_ids.shape[1]/2):].reshape(1, (input_ids.shape[1]-math.ceil(input_ids.shape[1]/2)) * input_ids.shape[-1])
                masked_mask[1][:(mask.shape[1]-math.ceil(mask.shape[1]/2)) * mask.shape[-1]] = \
                    mask[i][math.ceil(mask.shape[1]/2):].reshape(1, (mask.shape[1]-math.ceil(mask.shape[1]/2)) * mask.shape[-1])

            longformer_emb = self.backbone(masked_ids, attention_mask=masked_mask)[0]
            longfomer_no_pad_list.append(longformer_emb.reshape(-1, longformer_emb.shape[-1]))
            if self.args.speaker != 'text':
                speaker_ids_no_pad_list.append(speaker_ids_r[i][mask_r[i]==1])
        # input_ids_r = input_ids_r.narrow(-1, 0, max(sum_text_len))
        # mask_r = mask_r.narrow(-1, 0, max(sum_text_len))
        # longformer_emb = self.backbone(input_ids.reshape(-1, input_ids.shape[-1]), attention_mask=mask.reshape(-1, mask.shape[-1]))[0]  # Getting representation for each token in the text

        # longformer_emb_size = longformer_emb.shape[-1]
        # # # filter out masked tokens
        # start_ind = 0
        # longfomer_no_pad_list = []
        # speaker_ids_no_pad_list = []
        # for i in range(bs):
        #     end_ind = start_ind + input_ids.shape[1]
        #     longfomer_no_pad_list.append(torch.masked_select(longformer_emb[start_ind:end_ind], mask[i].unsqueeze(-1)==1).reshape(-1, longformer_emb_size))
        #     speaker_ids_no_pad_list.append(torch.masked_select(speaker_ids[i], mask[i].unsqueeze(-1)==1).reshape(-1, speaker_ids[0].shape[-1]))
        #     start_ind = end_ind

        if self.args.random_queries:
            raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 1) * self.query_sigma + self.query_mu #raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 0.5)
        else:
            raw_query_embed = self.query_embed.weight

        if not self.args.use_gold_mentions:
            hs, memory = self.transformer(self.input_proj(longfomer_no_pad_list), mask, raw_query_embed) # [dec_layers, 1, num_queries, emb], [1, seg*seq, emb]
        else:
            span_starts = [torch.tensor([m[0] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_ends = [torch.tensor([m[1] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_emb, span_mask, avg_speaker_onehot = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends, num_mentions, speaker_ids_no_pad_list, genre)  # [mentions, emb']
            span_emb = self.span_proj(span_emb) # [mentions, emb]
            if self.args.speaker == 'after':
                span_emb = torch.cat([span_emb, avg_speaker_onehot], 2)
            hs, memory = self.transformer(span_emb, span_mask, raw_query_embed, gold_matrix, cluster_number)  # [dec_layers, bs, num_queries, emb], [bs, mentions, emb]


        last_hs = hs[-1] # [1, num_queries, emb]
        memory = memory[0][:-1].unsqueeze(0)
        cluster_logits, coref_logits, mention_logits = self.calc_cluster_and_coref_logits(last_hs, memory, gold_mentions is not None, span_mask, gold_mentions.shape[1])

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

    def generate(self, input_ids, sum_text_len, mask, gold_mentions, num_mentions, speaker_ids, genre, threshold, gold_mentions_list):
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
        input_ids_r = input_ids.reshape(input_ids.shape[0], -1)
        mask_r = mask.reshape(mask.shape[0], -1)
        if self.args.speaker != 'text':
            speaker_ids_r = speaker_ids.reshape(speaker_ids.shape[0], -1, speaker_ids.shape[-1])
        longfomer_no_pad_list = []
        speaker_ids_no_pad_list = []
        for i in range(input_ids_r.shape[0]):
            masked_ids = input_ids_r[i][mask_r[i]==1].unsqueeze(0)
            masked_mask = torch.ones_like(masked_ids).unsqueeze(0)
            if masked_ids.shape[-1] > self.args.max_seq_length:
                masked_ids = torch.zeros([2, math.ceil(input_ids.shape[1]/2) * input_ids.shape[-1]], dtype=torch.long)
                masked_mask = torch.zeros([2, math.ceil(mask.shape[1]/2) * mask.shape[-1]], dtype=torch.long)
                masked_ids[0] = input_ids[i][:math.ceil(input_ids.shape[1]/2)].reshape(1, math.ceil(input_ids.shape[1]/2) * input_ids.shape[-1])
                masked_mask[0] = mask[i][:math.ceil(mask.shape[1]/2)].reshape(1, math.ceil(mask.shape[1]/2) * mask.shape[-1])
                masked_ids[1][:(input_ids.shape[1]-math.ceil(input_ids.shape[1]/2)) * input_ids.shape[-1]] = \
                    input_ids[i][math.ceil(input_ids.shape[1]/2):].reshape(1, (input_ids.shape[1]-math.ceil(input_ids.shape[1]/2)) * input_ids.shape[-1])
                masked_mask[1][:(mask.shape[1]-math.ceil(mask.shape[1]/2)) * mask.shape[-1]] = \
                    mask[i][math.ceil(mask.shape[1]/2):].reshape(1, (mask.shape[1]-math.ceil(mask.shape[1]/2)) * mask.shape[-1])

            longformer_emb = self.backbone(masked_ids, attention_mask=masked_mask)[0]
            longfomer_no_pad_list.append(longformer_emb.reshape(-1, longformer_emb.shape[-1]))
            if self.args.speaker != 'text':
                speaker_ids_no_pad_list.append(speaker_ids_r[i][mask_r[i]==1])
        # input_ids_r = input_ids_r.narrow(-1, 0, max(sum_text_len))
        # mask_r = mask_r.narrow(-1, 0, max(sum_text_len))
        # longformer_emb = self.backbone(input_ids.reshape(-1, input_ids.shape[-1]), attention_mask=mask.reshape(-1, mask.shape[-1]))[0]  # Getting representation for each token in the text

        # longformer_emb_size = longformer_emb.shape[-1]
        # # # filter out masked tokens
        # start_ind = 0
        # longfomer_no_pad_list = []
        # speaker_ids_no_pad_list = []
        # for i in range(bs):
        #     end_ind = start_ind + input_ids.shape[1]
        #     longfomer_no_pad_list.append(torch.masked_select(longformer_emb[start_ind:end_ind], mask[i].unsqueeze(-1)==1).reshape(-1, longformer_emb_size))
        #     speaker_ids_no_pad_list.append(torch.masked_select(speaker_ids[i], mask[i].unsqueeze(-1)==1).reshape(-1, speaker_ids[0].shape[-1]))
        #     start_ind = end_ind

        if self.args.random_queries:
            raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 1) * self.query_sigma + self.query_mu #raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 0.5)
        else:
            raw_query_embed = self.query_embed.weight

        if not self.args.use_gold_mentions:
            hs, memory = self.transformer(self.input_proj(longfomer_no_pad_list), mask, raw_query_embed) # [dec_layers, 1, num_queries, emb], [1, seg*seq, emb]
        else:
            span_starts = [torch.tensor([m[0] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_ends = [torch.tensor([m[1] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_emb, span_mask, avg_speaker_onehot = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends, num_mentions, speaker_ids_no_pad_list, genre)  # [mentions, emb']
            span_emb = self.span_proj(span_emb) # [mentions, emb]
            if self.args.speaker == 'after':
                span_emb = torch.cat([span_emb, avg_speaker_onehot], 2)
            cluster_logits, coref_logits, predicted_clusters  = self.transformer.generate(span_emb, span_mask, raw_query_embed, self.is_cluster, span_mask, self.IO_score, threshold, gold_mentions_list)  # [dec_layers, bs, num_queries, emb], [bs, mentions, emb]


        # last_hs = hs[-1] # [1, num_queries, emb]
        # cluster_logits, coref_logits, mention_logits = self.calc_cluster_and_coref_logits(last_hs, memory, gold_mentions is not None, span_mask, gold_mentions.shape[1])

        # # aux_coref_logits = [self.calc_cluster_and_coref_logits(curr_hs, memory)[1] for curr_hs in hs[:-1]]

        # # coref_logits = self.temp_embed.weight[:, :doc_len].unsqueeze(0).sigmoid()
        # # cluster_logits = self.temp_cluster_embed.weight.unsqueeze(0).sigmoid()
        # # coref_logits = coref_logits * cluster_logits

        # # if self.aux_loss:
        # #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # out = {"coref_logits": coref_logits,
        #         "cluster_logits": cluster_logits,
        #         "mention_logits": mention_logits}
        #         # "aux_coref_logits": aux_coref_logits}
        return cluster_logits, coref_logits, predicted_clusters

    def make_batch_same_len(self, input_ids, mask, sum_text_len):
        input_ids_pads = torch.ones(1, self.args.max_segment_len, dtype=torch.int, device=input_ids[0].device) * TOKENS_PAD
        mask_pads = torch.zeros(1, self.args.max_segment_len, dtype=torch.int, device=input_ids[0].device)

        max_seq_num = np.argmax(sum_text_len)
        seq_num = input_ids[max_seq_num].shape[0]
        #TODO: change to martix mult and sum
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

    def calc_cluster_and_coref_logits(self, last_hs, memory, is_gold_mention, span_mask, max_num_mentions):
        # last_hs [bs, num_queries, emb]
        # memory [bs, tokens, emb]

        cluster_logits = self.is_cluster(last_hs).sigmoid()  # [bs, num_queries, 1]
        if self.args.add_junk:
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
            coref_logits.append(torch.cat([cur_coref_logits, (torch.ones(1, cur_coref_logits.shape[1], max_num_mentions-cur_coref_logits.shape[2]) * -1).to(cur_coref_logits.device)], dim=2))

            if self.args.add_junk:
                mention_logits_masked.append(torch.cat([mention_logits[i][span_mask[i]==1].unsqueeze(0), (torch.ones(1, max_num_mentions-cur_coref_logits.shape[2], 1) * -1).to(mention_logits.device)], dim=1))
        # if not is_gold_mention:  #TODO: do I want this?
        #     coref_logits = coref_logits * cluster_logits

        if self.args.add_junk:
            mention_logits_masked = torch.cat(mention_logits_masked)

        return cluster_logits, torch.cat(coref_logits), mention_logits_masked

    def get_span_emb(self, context_outputs_list, span_starts, span_ends, num_mentions, speaker_ids_masked, genre):
        max_mentions = num_mentions.max()
        span_mask_list = []
        span_emb_list = []
        avg_speaker_onehot_list = []
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
            if self.args.speaker == 'before' or self.args.speaker == 'after':
                avg_speaker_onehot = []
                for j in range(num_mentions[i]):
                    avg_speaker_onehot.append((speaker_ids_masked[i][span_starts[i][j]:span_ends[i][j]+1].sum(0) / (span_ends[i][j]-span_starts[i][j]+1.0)).unsqueeze(0))
                avg_speaker_onehot = torch.cat(avg_speaker_onehot,0)
                if self.args.speaker == 'before':
                    span_emb_construct.append(avg_speaker_onehot)
                avg_speaker_onehot_list.append(torch.cat([avg_speaker_onehot, torch.zeros(max_mentions-avg_speaker_onehot.shape[0], avg_speaker_onehot.shape[1], dtype=torch.float, device=context_outputs_list[i].device)]).unsqueeze(0))
            span_emb_cat = torch.cat(span_emb_construct, 1)
            span_mask = torch.cat([torch.ones(span_emb_cat.shape[0], dtype=torch.int, device=context_outputs_list[i].device), \
                torch.zeros(max_mentions-span_emb_cat.shape[0], dtype=torch.int, device=context_outputs_list[i].device)])
            span_emb_cat = torch.cat([span_emb_cat, torch.zeros(max_mentions-span_emb_cat.shape[0], span_emb_cat.shape[1], dtype=torch.float, device=context_outputs_list[i].device)])

            span_emb_list.append(span_emb_cat.unsqueeze(0))  
            span_mask_list.append(span_mask.unsqueeze(0))  
        span_emb_tensor = torch.cat(span_emb_list, 0)
        span_mask_tensor = torch.cat(span_mask_list, 0)
        if len(avg_speaker_onehot_list) > 0:
            avg_speaker_onehot_list = torch.cat(avg_speaker_onehot_list, 0)
        return span_emb_tensor, span_mask_tensor, avg_speaker_onehot_list  # [k, emb], [K, T]

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


class MatchingLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, training_matcher, eval_matcher, eos_coef, cost_is_cluster, cost_coref, cost_is_mention, args):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.training_matcher = training_matcher
        self.eval_matcher = eval_matcher
        self.cost_is_cluster = cost_is_cluster
        self.cost_coref = cost_coref
        self.cost_is_mention = cost_is_mention
        self.args = args
        self.eos_coef = eos_coef


    def forward(self, outputs, targets, is_training=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if is_training:
            matched_predicted_cluster_id, matched_gold_cluster_id = self.training_matcher(outputs, targets)
        else:
            matched_predicted_cluster_id, matched_gold_cluster_id = self.eval_matcher(outputs, targets)

        targets_clusters = targets['clusters']
        targets_mentions = targets['mentions']
        bs = outputs["coref_logits"].shape[0]
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[]}
        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i]
            if coref_logits.shape[0] > 1:
                coref_logits = coref_logits.squeeze(0)  # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i].squeeze(0) # [num_queries]
            if len(cluster_logits.shape) > 1:
                cluster_logits = cluster_logits.squeeze(0)
            if self.args.add_junk:
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
                
            if not self.args.add_junk or sum(targets_mentions[i].shape) == 0:
                cost_is_mention = torch.tensor(0)
            else:
                if sum(mention_logits.shape) == 0:
                    mention_logits = mention_logits.reshape(1)
                else:
                    mention_logits = mention_logits[:targets_mentions[i].shape[0]]
                weight_mention = targets_mentions[i] + self.eos_coef * (1 - targets_mentions[i])
                cost_is_mention = F.binary_cross_entropy(mention_logits, targets_mentions[i], weight=weight_mention)

            coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, targets_clusters[i].shape[1]).to(coref_logits.device))

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

    training_matcher = build_matcher(args, "Ordered")
    eval_matcher = build_matcher(args, "Hungarian")
    # TODO maybe return consideration of aux loss

    criterion = MatchingLoss(training_matcher=training_matcher, eval_matcher=eval_matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_is_mention=args.cost_is_mention,
                             cost_coref=args.cost_coref, args=args)

    # if args.loss == 'match':
    #     criterion = MatchingLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
    # elif args.loss == 'bcubed':
    #     criterion = BCubedLoss()

    criterion.to(device)
    # postprocessors = {'bbox': PostProcess()}

    return model, criterion #, postprocessors
