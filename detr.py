# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import logging
import os
import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, Tensor
from consts import OUT_KEYS, TOKENS_PAD, LETTERS_LIST
from tqdm import tqdm
import math
import pytorch_lightning as pl
from metrics import CorefEvaluator
import shutil
from transformers import AutoTokenizer
from toy_data import get_toy_data_objects

from optimization import WarmupLinearSchedule, WarmupExponentialSchedule
import numpy as np

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)
#
# from .backbone import build_backbone
from matcher import build_matcher
from transformer import build_transformer
from transformers import LongformerConfig, LongformerModel
from typing import List
from utils import calc_predicted_clusters, create_gold_matrix, tensor_and_remove_empty, create_junk_gold_mentions, save_checkpoint
from transformers import AutoConfig, CONFIG_MAPPING
from misc import NestedTensor, accuracy
from consts import GENRES
from eval import print_predictions, error_analysis
from data import get_data_objects

logger = logging.getLogger(__name__)

class DETRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval_loader = None
        self.train_loader = None
        self.tokenizer = None

    def setup(self, stage):
        self.len_train_loader = len(self.train_dataloader())
                                
        if self.args.max_steps > 0:
            self.args.t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (self.len_train_loader // self.args.gradient_accumulation_steps) + 1
        else:
            self.args.t_total = self.len_train_loader // self.args.gradient_accumulation_steps * self.args.num_train_epochs
            self.args.warmup_steps = self.args.warmup_steps // self.args.gradient_accumulation_steps

        if self.args.train_batch_size > 1:
            self.args.eval_steps = -1 if self.args.eval_steps == -1 else max(1, int(round(self.args.eval_steps / self.args.train_batch_size)))
            self.args.save_steps = -1 if self.args.save_steps == -1 else max(1, int(round(self.args.save_steps / self.args.train_batch_size)))
            self.args.logging_steps = -1 if self.args.logging_steps == -1 else max(1, int(round(self.args.logging_steps / self.args.train_batch_size)))


    def val_dataloader(self):
        if self.eval_loader is None:
            if self.args.input_type == 'ontonotes':
                self.eval_dataset, eval_sampler, self.eval_loader, self.args.eval_batch_size = get_data_objects(self.args, self.args.predict_file, False)
            else:
                self.args.eval_batch_size = 1
                self.eval_dataset, self.eval_loader = get_toy_data_objects(self.args.input_type.split('_')[0], False, self.args, 0.1, int(self.args.input_type.split('_')[1]))

        return self.eval_loader

    def train_dataloader(self):
        if self.train_loader is None:
            if self.args.input_type == 'ontonotes':
                train_dataset, train_sampler, self.train_loader, self.args.train_batch_size = get_data_objects(self.args, self.args.train_file, True)
            else:
                self.args.train_batch_size = 1
                train_dataset, self.train_loader = get_toy_data_objects(self.args.input_type.split('_')[0], True, self.args, 0.9, int(self.args.input_type.split('_')[1]))
        return self.train_loader


class DETR(pl.LightningModule):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, criterion, transformer, num_queries, args, aux_loss=False):
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
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.is_cluster = nn.Linear(hidden_dim, 1)
        self.backbone = backbone
        self.criterion = criterion
        self.aux_loss = aux_loss
        self.args = args
        self.threshold = self.args.threshold if self.args.threshold > 0 else 0.5
        self.same_thresh_count = 0
        self.thresh_delta = 0.2
        self.train_evaluator = CorefEvaluator()
        self.eval_evaluator = CorefEvaluator()
        if args.single_distribution_queries:
            self.query_mu = nn.Parameter(torch.randn(1, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(1, hidden_dim))
        else:
            self.query_mu = nn.Parameter(torch.randn(num_queries, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.input_proj = nn.Linear(backbone.config.hidden_size, hidden_dim)
        self.word_attn_projection = nn.Linear(backbone.config.hidden_size, 1)
        self.span_proj = nn.Linear(3*backbone.config.hidden_size+20, hidden_dim) # TODO config
        self.span_width_embed = nn.Embedding(30, 20)
             
        self.mention_classifier = nn.Linear(hidden_dim, 1)

        self.IO_score = nn.Sequential(
            nn.Linear(2*hidden_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, self.args.BIO),
        ) #query and token concatenated, resulting in IO score

        self.query_head = nn.Linear(hidden_dim, 75)
        self.token_head = nn.Linear(hidden_dim, 75)

        self.input_ids_pads = torch.ones(1, self.args.max_segment_len, dtype=torch.int, device=self.args.device) * TOKENS_PAD
        self.mask_pads = torch.zeros(1, self.args.max_segment_len, dtype=torch.int, device=self.args.device)
        self.recent_train_losses = []
        self.recent_train_losses_parts = {}
        self.losses = []
        self.losses_parts = {}
        self.batch_sizes = []
        self.all_cluster_logits = []
        self.all_coref_logits = []
        self.all_gold_mentions = []
        self.all_predicted_clusters = []
        self.all_input_ids = []
        self.all_gold_clusters = []
        if self.args.input_type.split('_')[0] == 'letters':
            self.query_cluster_confusion_matrix = np.zeros([args.num_queries, 26], dtype=int)
        else:
            self.query_cluster_confusion_matrix = np.zeros([args.num_queries, args.num_queries], dtype=int)

        self.best_f1 = 0
        self.best_f1_epoch = -1
        self.epoch = 0
        self.step_num = 0

        if self.args.slots:  
            dim = args.hidden_dim      
            self.num_slots = num_queries
            self.iters = 6
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
                nn.Linear(dim, dim*2),
                nn.ReLU(inplace = True),
                nn.Linear(dim*2, dim)
            )

            self.norm_input  = nn.LayerNorm(dim)
            self.norm_slots  = nn.LayerNorm(dim)
            self.norm_pre_ff = nn.LayerNorm(dim)
                
            self.mlp_classifier = nn.Sequential(
                nn.Linear(dim, int(dim/2)),
                nn.ReLU(inplace = True),
                nn.Linear(int(dim/2), 1),
                nn.Sigmoid()
            )

        if self.args.resume_from:
            global_step = self.args.resume_from.rstrip('/').split('-')[-1]
            checkpoint = torch.load(self.args.resume_from + '/model.step-' + global_step + '.pt', map_location=args.device)
            model_to_load = self.module if hasattr(self, 'module') else self  # Take care of distributed/parallel training
            model_to_load.load_state_dict(checkpoint['model'])
            x=1

    def forward(self, input_ids, mask, gold_mentions, num_mentions):
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
        if self.args.slots:
            raw_query_embed = None
        elif self.args.random_queries:
            raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 1) * self.query_sigma + self.query_mu #raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 0.5)
        else:
            raw_query_embed = self.query_embed.weight

        bs = input_ids.shape[0]
        input_ids_r = input_ids.reshape(input_ids.shape[0], -1)
        mask_r = mask.reshape(mask.shape[0], -1)
        longfomer_no_pad_list = []
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
            self.args.is_encoding = False

        if not self.args.use_gold_mentions:
            longfomer_no_pad = self.input_proj(torch.stack(longfomer_no_pad_list,0))
            span_mask = masked_mask.reshape(longfomer_no_pad.shape[:-1]) 
            hs, memory = self.transformer(longfomer_no_pad, span_mask, raw_query_embed, self.is_cluster, self.IO_score, self.args.cluster_block, self.args.is_encoding) # [dec_layers, 1, num_queries, emb], [1, seg*seq, emb]
        else:   #TODO: not good for sequences because only takes first and last letters and doesnt have a representation of the surrounding
            span_starts = [torch.tensor([m[0] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_ends = [torch.tensor([m[1] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_emb, span_mask = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends, num_mentions)  # [mentions, emb']
            span_emb = self.span_proj(span_emb) # [mentions, emb]
            hs, memory = self.transformer(span_emb, span_mask, raw_query_embed, self.is_cluster, self.IO_score, self.args.cluster_block, self.args.is_encoding)  # [dec_layers, bs, num_queries, emb], [bs, mentions, emb]

        if not self.args.slots:
            last_hs = hs[-1] # [bs, num_queries, emb]
            cluster_logits, coref_logits, mention_logits = self.calc_cluster_and_coref_logits(last_hs, memory, span_mask)

        out = {"coref_logits": coref_logits,
                "cluster_logits": cluster_logits,
                "mention_logits": mention_logits,
                "memory": memory}
        return out

    def slot_attention(self, input_emb):
        bs, doc_len, emb, device = *input_emb.shape, input_emb.device
        
        mu = self.slots_mu.expand(bs, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(bs, self.num_slots, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

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

        return cluster_logits, coref_logits, torch.tensor([])

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.args.lr_backbone, #TODO: learn how to freeze backbone
            },
        ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr, weight_decay=self.args.weight_decay)

        lr_scheduler = {
            'scheduler': WarmupLinearSchedule(self.optimizer, warmup_steps=self.args.warmup_steps, t_total=self.args.t_total),
            "interval": "step"
            }
        return [self.optimizer], [lr_scheduler]
        

    def training_step(self, batch, batch_idx):
        if self.args.do_train:
            sum_text_len = [sum(tl) for tl in batch['text_len']]
            gold_clusters = batch['clusters']

            gold_mentions_list = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
            if self.args.add_junk:
                if self.args.input_type == 'ontonotes':
                    gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device)
                else:
                    gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device, 0)
            else:
                gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=self.args.device) for gm in gold_mentions_list]

            input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = \
                tensor_and_remove_empty(batch, gold_mentions_list, self.args, self.input_ids_pads, self.mask_pads)
            if len(input_ids) == 0:
                return 0

            gold_matrix = create_gold_matrix(self.args.device, sum_text_len, self.args.num_queries, gold_clusters, gold_mentions_list, self.args.use_gold_mentions, self.args.BIO)

            outputs = self(input_ids, input_mask, gold_mentions, num_mentions)
            cluster_logits, coref_logits, mention_logits = outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

            if self.args.add_junk:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
                                                            self.threshold, gold_mentions_list, self.args.use_gold_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
            else:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                            self.threshold, gold_mentions_list, self.args.use_gold_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
            self.train_evaluator.update(predicted_clusters, gold_clusters)
            loss, loss_parts = self.criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})
            
            self.recent_train_losses.append(loss.item())
            for key in loss_parts.keys():
                if key in self.recent_train_losses_parts.keys() and len(self.recent_train_losses_parts[key]) > 0:
                    self.recent_train_losses_parts[key] += loss_parts[key]
                else:
                    self.recent_train_losses_parts[key] = loss_parts[key]

            if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and batch_idx % self.args.logging_steps == 0:
                self.trainer.logger.log_metrics({'lr': self.optimizer.param_groups[0]['lr']}, self.step_num)
                self.trainer.logger.log_metrics({'lr_bert': self.optimizer.param_groups[1]['lr']}, self.step_num)
                self.trainer.logger.log_metrics({'loss': np.mean(self.recent_train_losses)}, self.step_num)
                for key in self.recent_train_losses_parts.keys():
                    self.trainer.logger.log_metrics({key: np.mean(self.recent_train_losses_parts[key])}, self.step_num)
                self.recent_train_losses.clear()
                self.recent_train_losses_parts.clear()

            self.step_num += 1

            return {'loss': loss}

    def training_epoch_end(self, train_step_outputs):
        self.log('epoch', self.epoch)
        if self.args.do_train:
            t_p, t_r, t_f1 = self.train_evaluator.get_prf()
            if self.args.local_rank in [-1, 0]:
                self.trainer.logger.log_metrics({'Train Precision': t_p}, self.step_num)
                self.trainer.logger.log_metrics({'Train Recall': t_r}, self.step_num)
                self.trainer.logger.log_metrics({'Train F1': t_f1}, self.step_num)
                logger.info(f'Train f1 {t_f1}, precision {t_p} , recall {t_r}')

            self.recent_train_losses.clear()
            self.recent_train_losses_parts.clear()
            self.train_evaluator = CorefEvaluator()
            self.epoch += 1

    def validation_step(self, batch, batch_idx):
        sum_text_len = [sum(tl) for tl in batch['text_len']]
        gold_clusters = batch['clusters']

        gold_mentions_list = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
        if self.args.add_junk:
            if self.args.input_type == 'ontonotes':
                gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device)
            else:
                gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device, 0)
        else:
            gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=self.args.device) for gm in gold_mentions_list]
        
        gold_matrix = create_gold_matrix(self.args.device, sum_text_len, self.args.num_queries, gold_clusters, gold_mentions_list, self.args.use_gold_mentions, self.args.BIO)

        input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = \
            tensor_and_remove_empty(batch, gold_mentions_list, self.args, self.input_ids_pads, self.mask_pads)
        if len(input_ids) == 0:
            return 0

        outputs = self(input_ids, input_mask, gold_mentions, num_mentions)
        cluster_logits, coref_logits, mention_logits = outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']
        targets = {'clusters':gold_matrix, 'mentions':gold_mentions_vector}
        matched_predicted_cluster_id_real, matched_gold_cluster_id_real, _, _ = self.criterion.matcher(outputs, targets)
        if matched_gold_cluster_id_real[0] is not False:
            if self.args.input_type.split('_')[0] != 'letters':
                for i, j in zip(matched_predicted_cluster_id_real[0], matched_gold_cluster_id_real[0]):
                    self.query_cluster_confusion_matrix[i][j] += 1
            else:
                mention_ids = input_ids[0][0][[[m[0] for i in range(len(gold_mentions)) for m in gold_mentions[i]]]]
                for i, j in zip(matched_predicted_cluster_id_real[0], matched_gold_cluster_id_real[0]):
                    target_letters = [self.toy_onehot_dict[l] for l in mention_ids[targets['clusters'][0][j]==1].cpu().numpy()]
                    for t in target_letters:
                        self.query_cluster_confusion_matrix[i][t] += 1

        # if self.args.add_junk:
        #     predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
        #                                                 self.threshold, gold_mentions_list, self.args.use_gold_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
        # else:
        #     predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
        #                                                 self.threshold, gold_mentions_list, self.args.use_gold_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
        # self.eval_evaluator.update(predicted_clusters, gold_clusters)
        loss, loss_parts = self.criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})
        self.losses.append(loss.mean().detach().cpu())
        for key in loss_parts.keys():
            if key in self.losses_parts.keys() and len(self.losses_parts[key]) > 0:
                self.losses_parts[key] += loss_parts[key]
            else:
                self.losses_parts[key] = loss_parts[key]
        self.batch_sizes.append(loss.shape[0]) 

        # self.all_predicted_clusters += predicted_clusters  
        
        self.all_input_ids += input_ids    
        self.all_gold_clusters += gold_clusters
        self.all_cluster_logits += cluster_logits.detach().cpu()
        self.all_coref_logits += coref_logits.detach().cpu()
        self.all_gold_mentions += gold_mentions_list

        return {'loss': loss}

    def eval_by_thresh(self, threshold):
        evaluator = CorefEvaluator()
        all_predicted_clusters = []
        metrics = [0] * 5    
        for i, (cluster_logits, coref_logits, gold_clusters, gold_mentions) in enumerate(
                zip(self.all_cluster_logits, self.all_coref_logits, self.all_gold_clusters, self.all_gold_mentions)):                
            predicted_clusters = calc_predicted_clusters(cluster_logits.unsqueeze(0), coref_logits.unsqueeze(0), [], \
                threshold, [gold_mentions], self.args.use_gold_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
            all_predicted_clusters += predicted_clusters
            evaluator.update(predicted_clusters, [gold_clusters])
        p, r, f1 = evaluator.get_prf()
        return p, r, f1, metrics, all_predicted_clusters

    def validation_epoch_end(self, val_step_outputs):
        eval_loss = np.average(self.losses, weights=self.batch_sizes)
        losses_parts = {key:np.average(self.losses_parts[key]) for key in self.losses_parts.keys()}

        if self.args.threshold > 0 or (self.thresh_delta < 0.2 and self.same_thresh_count > 3) or self.epoch == 0:
            p, r, f1, best_metrics, self.all_predicted_clusters = self.eval_by_thresh(self.threshold)
        else:
            if self.thresh_delta == 0.2:
                thresh_start = 0.05
                thresh_end = 1
            else:
                thresh_start = max(0.01, self.threshold - 2*self.thresh_delta)
                thresh_end = min(1, self.threshold + 2.5*self.thresh_delta)
                
            best = [-1, -1, -1]
            best_metrics = []
            best_threshold = None
            for threshold in tqdm(np.arange(thresh_start, thresh_end, self.thresh_delta), desc='Searching for best threshold'):
                p, r, f1, metrics, all_predicted_clusters = self.eval_by_thresh(threshold)
                if f1 > best[-1]:
                    best = p,r,f1
                    best_metrics = metrics
                    best_threshold = threshold
                    self.all_predicted_clusters = all_predicted_clusters
            p,r,f1 = best
            if best_threshold == self.threshold:
                self.same_thresh_count += 1
                if self.same_thresh_count == 3 and self.thresh_delta == 0.2:
                    self.thresh_delta = 0.02
                    self.same_thresh_count = 0
            else:
                self.same_thresh_count = 0
            self.threshold = best_threshold

        print_predictions(self.all_predicted_clusters, self.all_gold_clusters, self.all_input_ids, self.args, self.tokenizer)
        prec_gold_to_one_pred, prec_pred_to_one_gold, avg_gold_split_without_perfect, avg_gold_split_with_perfect, \
            avg_pred_split_without_perfect, avg_pred_split_with_perfect, prec_biggest_gold_in_pred_without_perfect, \
                prec_biggest_gold_in_pred_with_perfect, prec_biggest_pred_in_gold_without_perfect, prec_biggest_pred_in_gold_with_perfect = \
                    error_analysis(self.all_predicted_clusters, self.all_gold_clusters)
    
        non_zero_rows = np.where(np.sum(self.query_cluster_confusion_matrix, 1) > 0)[0]
        non_zero_cols = np.where(np.sum(self.query_cluster_confusion_matrix, 0) > 0)[0]
        print('rows - predict, cols - gt')
        line_to_print = '   '
        for i in non_zero_cols:
            line_to_print += str(i) + ' ' + ('' if i>=10 else ' ')
        print(line_to_print)
        for i in non_zero_rows:
            line_to_print = str(i) + ' ' + ('' if i>=10 else ' ')
            for j in non_zero_cols:
                line_to_print += str(self.query_cluster_confusion_matrix[i][j]) + ' ' + ('' if self.query_cluster_confusion_matrix[i][j]>=10 else ' ')
            print(line_to_print)
    

        results = {'loss': eval_loss,
                'avg_f1': f1,
                'threshold': self.threshold,
                'precision': p,
                'recall': r,  
                'prec_gold_to_one_pred': prec_gold_to_one_pred,  
                'prec_pred_to_one_gold': prec_pred_to_one_gold,  
                'avg_gold_split_without_perfect': avg_gold_split_without_perfect,  
                'avg_gold_split_with_perfect': avg_gold_split_with_perfect,  
                'avg_pred_split_without_perfect': avg_pred_split_without_perfect,  
                'avg_pred_split_with_perfect': avg_pred_split_with_perfect,  
                'prec_biggest_gold_in_pred_without_perfect': prec_biggest_gold_in_pred_without_perfect, 
                'prec_biggest_gold_in_pred_with_perfect': prec_biggest_gold_in_pred_with_perfect,  
                'prec_biggest_pred_in_gold_without_perfect': prec_biggest_pred_in_gold_without_perfect,  
                'prec_biggest_pred_in_gold_with_perfect': prec_biggest_pred_in_gold_with_perfect,  
                'prec_correct_mentions': best_metrics[0],
                'prec_gold': best_metrics[1],
                'prec_junk': best_metrics[2],
                'prec_correct_gold_clusters': best_metrics[3],
                'prec_correct_predict_clusters': best_metrics[4]} | losses_parts

        for key, value in results.items():
            self.trainer.logger.log_metrics({'eval_{}'.format(key): value}, self.step_num)
        self.log('eval_avg_f1', torch.tensor(results['avg_f1']))

        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:

            def out(s):
                logger.info(str(s))
                writer.write(str(s) + '\n')

            out("***** Eval results {} *****".format(self.epoch))

            for key in sorted(results.keys()):
                out("eval %s = %s" % (key, str(results[key])))

        if self.step_num > 0 and self.args.save_epochs > 0 and (self.epoch + 1) % self.args.save_epochs == 0 or self.epoch + 1 == self.args.num_train_epochs:
            if f1 > self.best_f1:
                prev_best_f1 = self.best_f1
                prev_best_f1_epoch = self.best_f1_epoch
                output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.epoch))
                save_checkpoint(self.args, self.epoch, self.threshold, self, self.optimizer, output_dir)
                print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_epoch}')
                self.best_f1 = f1
                self.best_f1_epoch = self.epoch
                print(f'saved checkpoint with f1 {self.best_f1} in step {self.best_f1_epoch} to {output_dir}')
                if prev_best_f1_epoch > -1:
                    path_to_remove = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(prev_best_f1_epoch))
                    shutil.rmtree(path_to_remove)
                    print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')
            self.trainer.logger.log_metrics({'eval_best_f1': self.best_f1}, self.step_num)
            try:
                self.trainer.logger.log_metrics({'eval_best_f1_checkpoint': os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.best_f1_epoch))}, self.step_num)
            except:
                pass
        
        self.eval_evaluator = CorefEvaluator()
        self.recent_train_losses = []
        self.recent_train_losses_parts = {}
        self.losses = []
        self.losses_parts = {}
        self.batch_sizes = []
        self.all_cluster_logits = []
        self.all_coref_logits = []
        self.all_gold_mentions = []
        self.all_input_ids = []
        self.all_gold_clusters = []
        self.all_predicted_clusters = []
        self.query_cluster_confusion_matrix = np.zeros([self.args.num_queries, self.args.num_queries], dtype=int)
        # self.query_cluster_confusion_matrix = np.zeros([self.args.num_queries, 26], dtype=int)

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

    def calc_cluster_and_coref_logits(self, last_hs, memory, span_mask):
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
            coref_logits_unnorm = self.IO_score(coref_features) # [bs, num_queries, tokens/mentions, BIO(3/1)]

            if coref_logits_unnorm.shape[-1] > 1:
                cur_coref_logits = coref_logits_unnorm.softmax(-1)
            elif self.args.use_gold_mentions or self.args.softmax: 
                cur_coref_logits = coref_logits_unnorm.softmax(dim=1).squeeze(-1)
            else:
                cur_coref_logits = coref_logits_unnorm.sigmoid().squeeze(-1)
            if len(cur_coref_logits.shape) > 3:
                coref_logits.append(torch.cat([cur_coref_logits, (torch.ones(cur_coref_logits.shape[0], cur_coref_logits.shape[1], memory.shape[1]-cur_coref_logits.shape[2], cur_coref_logits.shape[3]) * -1).to(cur_coref_logits.device)], dim=2))
            else:
                coref_logits.append(torch.cat([cur_coref_logits, (torch.ones(cur_coref_logits.shape[0], cur_coref_logits.shape[1], memory.shape[1]-cur_coref_logits.shape[2]) * -1).to(cur_coref_logits.device)], dim=2))

            if self.args.add_junk:
                mention_logits_masked.append(torch.cat([mention_logits[i][span_mask[i]==1].unsqueeze(0), (torch.ones(1, memory.shape[1]-cur_coref_logits.shape[2], 1) * -1).to(mention_logits.device)], dim=1))
        # if not is_gold_mention:  #TODO: do I want this?
        #     coref_logits = coref_logits * cluster_logits

        if self.args.add_junk:
            mention_logits_masked = torch.cat(mention_logits_masked)

        return cluster_logits, torch.cat(coref_logits), mention_logits_masked

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
        matched_predicted_cluster_id_real, matched_gold_cluster_id_real, matched_predicted_cluster_id_junk, matched_gold_cluster_id_junk \
            = self.matcher(outputs, targets)

        targets_clusters = targets['clusters']
        targets_mentions = targets['mentions']
        bs = outputs["coref_logits"].shape[0]
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[], 'loss_embedding':[]}
        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries, tokens, BIO(3/1)]
            cluster_logits = outputs["cluster_logits"][i].squeeze() # [num_queries]
            real_token_target_cols = torch.sum(targets_clusters[i], -2) > 0

            if self.args.add_junk:
                mention_logits = outputs["mention_logits"][i].squeeze() # [tokens]
            #TODO: normalize according to number of clusters? (identical to DETR)

            # num_of_gold_clusters = len(targets)
            # num_of_gold_clusters = torch.as_tensor([num_of_gold_clusters], dtype=torch.float, device=coref_logits.device)
            # if is_dist_avail_and_initialized():
            #     torch.distributed.all_reduce(num_of_gold_clusters)
            # num_of_gold_clusters = torch.clamp(num_of_gold_clusters / get_world_size(), min=1).item()

            if self.args.is_cluster and not self.args.use_gold_mentions:
                gold_is_cluster = torch.zeros_like(cluster_logits)
                weight_cluster = self.eos_coef * torch.ones_like(cluster_logits)
                if matched_predicted_cluster_id_real[i] is not False:
                    gold_is_cluster[matched_predicted_cluster_id_real[i]] = 1
                    weight_cluster[matched_predicted_cluster_id_real[i]] = 1
                cost_is_cluster = F.binary_cross_entropy(cluster_logits, gold_is_cluster, weight=weight_cluster, reduction=self.args.reduction) / len(cluster_logits)
            else:
                cost_is_cluster = torch.tensor(0)
                
            if not self.args.add_junk or sum(targets_mentions[i].shape) == 0:
                cost_is_mention = torch.tensor(0)
            else:
                if sum(mention_logits.shape) == 0:
                    mention_logits = mention_logits.reshape(1)
                else:
                    mention_logits = mention_logits[:targets_mentions[i].shape[0]]
                weight_mention = targets_mentions[i] + self.eos_coef * (1 - targets_mentions[i])
                cost_is_mention = F.binary_cross_entropy(mention_logits, targets_mentions[i], weight=weight_mention, reduction=self.args.reduction) / len(mention_logits)

            coref_logits = torch.index_select(coref_logits, 1, torch.arange(0, targets_clusters[i].shape[1]).to(coref_logits.device))

            cost_coref = torch.tensor(0)
            embedding_loss = torch.tensor(0)
            if matched_predicted_cluster_id_real[i] is not False:
                permuted_coref_logits = coref_logits[torch.cat([matched_predicted_cluster_id_real[i],matched_predicted_cluster_id_junk[i]])]
                if not self.args.cluster_block and self.args.slots:
                    junk_cluster = torch.zeros_like(targets_clusters[i][matched_gold_cluster_id_junk[i]].transpose(0,1)[real_token_target_cols].transpose(0,1))
                    cost_coref = F.binary_cross_entropy(coref_logits[matched_predicted_cluster_id_real[i]], \
                        targets_clusters[i][matched_gold_cluster_id_real[i]], reduction=self.args.reduction) / coref_logits.shape[1] + \
                            F.binary_cross_entropy(coref_logits[matched_predicted_cluster_id_junk[i]].transpose(0,1)[real_token_target_cols].transpose(0,1), \
                        junk_cluster, reduction=self.args.reduction) / len(real_token_target_cols)
                else:
                    permuted_gold = targets_clusters[i][torch.cat([matched_gold_cluster_id_real[i],matched_gold_cluster_id_junk[i]])]
                    premuted_cluster_logits = cluster_logits[torch.cat([matched_predicted_cluster_id_real[i],matched_predicted_cluster_id_junk[i]])]

                    if self.args.cluster_block:
                        cost_coref = F.binary_cross_entropy(premuted_cluster_logits.unsqueeze(1)*permuted_coref_logits, permuted_gold, reduction=self.args.reduction) / coref_logits.shape[1]
                    # if self.args.BIO == 3:
                    #     cost_coref = F.cross_entropy(permuted_coref_logits.reshape([-1, 3]), permuted_gold.reshape([-1]), reduction=self.args.reduction) / coref_logits.shape[1]
                    else:
                        cost_coref = F.binary_cross_entropy(permuted_coref_logits, permuted_gold, reduction=self.args.reduction) / coref_logits.shape[1]

                if self.args.input_type.split('_')[0] == 'sequences':
                    memory = outputs["memory"][i].squeeze(0)
                    cluster_inds = targets_clusters[i][matched_gold_cluster_id_real[i]]
                    junk_cluster = 1 - torch.sum(cluster_inds, 0)
                    if torch.sum(junk_cluster) > 0:
                        cluster_inds = torch.cat([cluster_inds, junk_cluster.unsqueeze(0)])
                    avg_vector = torch.matmul(cluster_inds, memory) / torch.sum(cluster_inds, 1).reshape(-1, 1)
                    center_clusters_distances = torch.cdist(avg_vector, avg_vector)
                    diffs = 0
                    for x in range(cluster_inds.shape[0]):
                        diffs += torch.sqrt(torch.sum(torch.pow((memory - avg_vector[x]) * cluster_inds[x].unsqueeze(-1), 2))) / torch.sum(cluster_inds[x])
                    embedding_loss = (torch.max(torch.tensor(1, device=diffs.device),diffs)/cluster_inds.shape[0])\
                        / (torch.sum(center_clusters_distances)/(center_clusters_distances.shape[0]*center_clusters_distances.shape[1]))
                        
            elif coref_logits.shape[1] > 0:
                cost_coref = F.binary_cross_entropy(coref_logits, torch.zeros_like(coref_logits), reduction=self.args.reduction) / coref_logits.shape[1]

            if self.args.b3_loss:
                # b3_loss
                real_coref_logits = coref_logits
                # if self.args.is_cluster:
                #     real_coref_logits = coref_logits[]
                real_target_rows = targets_clusters[i][torch.sum(targets_clusters[i], 1) > 0]
                gold_predic_intersect = torch.pow(torch.matmul(real_target_rows, real_coref_logits.transpose(0,1)), 2)  # [gold_entities, predict_entities]  x[i, j] = \sum_k I[m_k \in e_i] * p[m_k \in e_j]
                r_num = torch.sum(torch.sum(gold_predic_intersect, 1) / torch.sum(real_target_rows, 1))
                r_den = torch.sum(real_target_rows)
                recall = torch.reshape(r_num / r_den, [])

                predict_gold_intersection = gold_predic_intersect.transpose(0, 1)
                p_num = torch.sum(torch.sum(predict_gold_intersection, 1) / torch.sum(coref_logits, 1))
                p_den = torch.sum(coref_logits)
                prec = torch.reshape(p_num / p_den, [])

                beta_2 = 2.0 ** 2
                f_beta = (1 + beta_2) * prec * recall / (beta_2 * prec + recall)

                cost_coref = 1. - f_beta

            costs_parts['loss_is_cluster'].append(self.cost_is_cluster * cost_is_cluster.detach().cpu())
            costs_parts['loss_is_mention'].append(self.cost_is_mention * cost_is_mention.detach().cpu())
            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            costs_parts['loss_embedding'].append(self.cost_coref * 5*embedding_loss.detach().cpu())
            total_cost = self.cost_coref * cost_coref + self.cost_is_cluster * cost_is_cluster + self.cost_is_mention * cost_is_mention + 5*self.cost_coref * embedding_loss
            costs.append(total_cost)
        return torch.stack(costs), costs_parts

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

    matcher = build_matcher(args)
    # TODO maybe return consideration of aux loss

    criterion = MatchingLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_is_mention=args.cost_is_mention,
                             cost_coref=args.cost_coref, args=args)
    criterion.to(device)

    model = DETR(
        backbone,
        criterion,
        transformer,
        num_queries=args.num_queries,
        args=args,
        aux_loss=args.aux_loss
    )

    # if args.loss == 'match':
    #     criterion = MatchingLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
    # elif args.loss == 'bcubed':
    #     criterion = BCubedLoss()

    # postprocessors = {'bbox': PostProcess()}

    return model #, postprocessors
