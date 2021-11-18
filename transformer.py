# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import numpy as np
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from utils import calc_predicted_clusters


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, gold_matrix, cluster_number, is_weighted_loss_smaller_mask, pos_embed=None):
        # flatten NxMxE to ExNxM
        bs, m, e = src.shape
        src = src.permute(1,0,2)
        if pos_embed is not None:
            pos_embed = pos_embed.transpose(0,1)
        binary_mask = mask == 0
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = query_embed
        memory = self.encoder(src, src_key_padding_mask=binary_mask, pos=pos_embed)

        gold_mask, memory, binary_mask, gold_matrix_permute = self.create_new_mask_mask_mentions(gold_matrix, cluster_number, memory, binary_mask, is_weighted_loss_smaller_mask)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=binary_mask, memory_mask=gold_mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory.transpose(0, 1), gold_matrix_permute, gold_mask

    def create_new_mask_mask_mentions(self, gold_matrix, cluster_number, memory, binary_mask, is_weighted_loss_smaller_mask):
        if is_weighted_loss_smaller_mask:
            factor = 0.5
        else:
            factor = 1
        idx = torch.arange(gold_matrix[0].shape[1], 0, -1, device = gold_matrix[0].device)
        tmp2 = gold_matrix[0] * idx
        indices = torch.argmax(tmp2, 1, keepdim=True).squeeze()
        indices = indices[:cluster_number]
        gold_matrix_sorted = torch.index_select(gold_matrix[0], 0, torch.argsort(indices))
        gold_matrix_sorted = torch.cat([gold_matrix_sorted, torch.zeros(gold_matrix_sorted.shape[0],1, device = gold_matrix[0].device)], 1)
        log_gold_matrix_sorted = torch.log(1 - factor*gold_matrix_sorted)
        gold_matrix_sumed = torch.cumsum(log_gold_matrix_sorted, axis=0)
        gold_mask = torch.cat([torch.zeros(1, log_gold_matrix_sorted.shape[1], device = gold_matrix[0].device), \
                            torch.index_select(gold_matrix_sumed, 0, torch.arange(log_gold_matrix_sorted.shape[0]-1, device = gold_matrix[0].device)), \
                            gold_matrix_sumed[-1] * torch.ones(gold_matrix[0].shape[0] - log_gold_matrix_sorted.shape[0], log_gold_matrix_sorted.shape[1], device = gold_matrix[0].device)], 0) 
        memory = torch.cat([memory, torch.ones(1, memory.shape[1], memory.shape[2], device = gold_matrix[0].device) * 0.001])
        binary_mask = torch.cat([binary_mask, torch.zeros(binary_mask.shape[0], 1, device = gold_matrix[0].device) == 1], 1)
        return gold_mask, memory, binary_mask, torch.argsort(indices).unsqueeze(0)

    def create_new_tgt_and_mask_concat_text_querys(self, tgt, src, memory, gold_matrix):
        # concat text and querys one after another. UNFINISHED
        text = src
        gold_matrix_cumsum = gold_matrix.cumsum()
        #TODO: sort goldmatrix by first mention
        new_tgt = []
        for i in range(tgt.shape[-1]):
            new_tgt.append(tgt[:,:,i])
            new_tgt.append(text[:,:,gold_matrix[i].ind])

    @torch.no_grad()
    def generate(self, src, mask, query_embed, is_cluster, span_mask, IO_score, threshold, gold_mentions_list, refeed_queries, predict_at_end, is_weighted_loss_smaller_mask, pos_embed=None):
        # flatten NxMxE to ExNxM
        bs, m, e = src.shape
        src = src.permute(1,0,2)
        if pos_embed is not None:
            pos_embed = pos_embed.transpose(0,1)
        binary_mask = mask == 0
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = query_embed
        memory = self.encoder(src, src_key_padding_mask=binary_mask, pos=pos_embed)
        cluster_logits, coref_logits, predicted_clusters  = self.decoder.mask_mention_decoder(tgt, memory, is_cluster, span_mask, IO_score, 
                                        threshold, gold_mentions_list, refeed_queries, predict_at_end, is_weighted_loss_smaller_mask, memory_key_padding_mask=binary_mask,
                                        pos=pos_embed, query_pos=query_embed)
        return cluster_logits, coref_logits, predicted_clusters 



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

    def mask_mention_decoder(self, tgt, memory, is_cluster, span_mask, IO_score, threshold, gold_mentions_list, refeed_queries, predict_at_end, is_weighted_loss_smaller_mask,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        if predict_at_end:
            refeed_queries = True
        cluster_logits = []
        coref_logits = []
        output = tgt[0].unsqueeze(0)
        cur_query_pos = query_pos[0].unsqueeze(0)
        memory_mask = torch.zeros([1, memory.shape[0]], device=output.device)
        i = 1

        while True:   #TODO i<100? mask not empty?
            for layer in self.layers:
                output = layer(output, memory, tgt_mask=tgt_mask, #TODO: will it work when output is not in query size?
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=cur_query_pos)

            if self.norm is not None:
                output = self.norm(output)
            tmp_memory = memory.transpose(0, 1)
            
            if predict_at_end:
                cur_output = output.transpose(0, 1)
                num_clusters = i
            else:
                cur_output = output[-1].unsqueeze(1)
                num_clusters = 1

            cur_cluster_logits, cur_coref_logits = self.create_logits(is_cluster, tmp_memory, span_mask, cur_output, IO_score, num_clusters)

            if not predict_at_end:
                coref_logits.append(cur_coref_logits)
                if is_cluster is not None:
                    cluster_logits.append(cur_cluster_logits)

            memory_mask = self.create_new_mask_mask_mentions(cur_coref_logits, memory_mask, refeed_queries, predict_at_end, is_weighted_loss_smaller_mask)

            if i >= tgt.shape[0] or sum(memory_mask[-1]) == memory_mask.shape[-1]:   #TODO: maybe it need to be i >= tgt.shape[0] but then there is a bug
                # num_of_empty_clusters = i - len(predicted_clusters[0])
                # if num_of_empty_clusters > 0:
                #     print(f'total of {num_of_empty_clusters} emptys out of {i}')
                if predict_at_end:
                    cluster_logits, coref_logits = self.create_logits(is_cluster, tmp_memory, span_mask, cur_output, IO_score, num_clusters)
                    if is_cluster is None:
                        predicted_clusters, _ = calc_predicted_clusters(None, coref_logits.cpu().detach(), [],
                                                                                threshold, gold_mentions_list, num_clusters=num_clusters)
                    else:
                        predicted_clusters, _ = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                                                threshold, gold_mentions_list, num_clusters=num_clusters)
                else:
                    coref_logits = torch.cat(coref_logits, 1)
                    if is_cluster is None:
                        cluster_logits = []
                        predicted_clusters, _ = calc_predicted_clusters(None, coref_logits.cpu().detach(), [],
                                                                                threshold, gold_mentions_list, num_clusters=num_clusters)
                    else:
                        cluster_logits = torch.cat(cluster_logits, 1)
                        predicted_clusters, _ = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                                                threshold, gold_mentions_list, num_clusters=num_clusters)
                return cluster_logits, coref_logits, predicted_clusters 

            i += 1
            if refeed_queries:
                output = tgt[:i]   #TODO: using the generated querys? the trained querys? different decisions for prediction and decoder's inputs? For now I think using the trained, because we wouldnt know the generated vectors in training
                cur_query_pos = query_pos[:i]
            else:
                output = tgt[i-1].unsqueeze(0)
                cur_query_pos = query_pos[i-1].unsqueeze(0)

    def create_logits(self, is_cluster, memory, span_mask, output, IO_score, num_queries):
        if is_cluster is None:
            cluster_logits = None
        else:
            cluster_logits = is_cluster(output).sigmoid()  # [bs, num_queries, 1]
        cur_memory = memory[0][span_mask[0]==1].unsqueeze(0)
        cur_last_hs = output
        num_tokens_or_mentions = cur_memory.shape[1]
        last_hs_tiled = cur_last_hs.unsqueeze(2).repeat(1, 1, num_tokens_or_mentions, 1) # [bs, num_queries, tokens/mentions, emb]
        memory_tiled = cur_memory.unsqueeze(1).repeat(1, num_queries, 1, 1) # [bs, num_queries, tokens/mentions, emb]
        coref_features = torch.cat([last_hs_tiled, memory_tiled], -1) # [bs, num_queries, tokens/mentions, 2 * emb]
        coref_logits_unnorm = IO_score(coref_features).squeeze(-1) # [bs, num_queries, tokens/mentions, 1]
        cur_coref_logits = coref_logits_unnorm.sigmoid()
        return cluster_logits, cur_coref_logits

    def create_new_mask_mask_mentions(self, coref_logits, memory_mask, refeed_queries, predict_at_end, is_weighted_loss_smaller_mask):
        if is_weighted_loss_smaller_mask:
            factor = 0.5
        else:
            factor = 1
        if predict_at_end:
            new_memory_mask = torch.zeros([memory_mask.shape[0]+1, memory_mask.shape[1]], device=memory_mask.device)
            for i in range(1, new_memory_mask.shape[0]):
                new_memory_mask[i] = new_memory_mask[i-1] + torch.log(1 - factor*coref_logits.squeeze()[i-1])
        else:
            added_mask = memory_mask[-1].clone()
            if len(coref_logits) > 0:
                added_mask += torch.log(1 - factor*coref_logits.squeeze())
            added_mask = added_mask.unsqueeze(0)
            if refeed_queries:
                new_memory_mask = torch.cat([memory_mask, added_mask], 0)   #TODO: first version: recreating the whole mask and not only the new cluster row.
            else:
                new_memory_mask = added_mask
        return new_memory_mask


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
