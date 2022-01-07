# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import pytorch_lightning as pl
import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from consts import OUT_KEYS, TOKENS_PAD
import math
import os
from tqdm import tqdm
import numpy as np
import shutil

from matcher import HungarianMatcher
from transformer import Transformer
from transformers import LongformerModel
from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING
from data import get_data_objects
from metrics import CorefEvaluator
from optimization import WarmupLinearSchedule
from coref_analysis import print_predictions, error_analysis
from utils import tensor_and_remove_empty, save_checkpoint, create_gold_matrix, calc_predicted_clusters, create_junk_gold_mentions

logger = logging.getLogger(__name__)


class DETRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval_loader = None
        self.train_loader = None

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
            self.eval_dataset, eval_sampler, self.eval_loader, self.args.eval_batch_size = get_data_objects(self.args, self.args.predict_file, False)
        return self.eval_loader

    def train_dataloader(self):
        if self.train_loader is None:
            train_dataset, train_sampler, self.train_loader, self.args.train_batch_size = get_data_objects(self.args, self.args.train_file, True)
        return self.train_loader

class DETR(pl.LightningModule):
    """ This is the DETR module that performs object detection """
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        device = torch.device(args.device)

        self.transformer = Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True
        )

        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        self.longformer = LongformerModel.from_pretrained(args.model_name_or_path,
                                            config=config,
                                            cache_dir=args.cache_dir)

        self.input_proj = nn.Linear(self.longformer.config.hidden_size, args.hidden_dim)
        self.span_proj = nn.Linear(3*self.longformer.config.hidden_size+20, args.hidden_dim) # TODO config
        self.word_attn_projection = nn.Linear(self.longformer.config.hidden_size, 1)
        self.span_width_embed = nn.Embedding(30, 20)

        matcher = HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, args=args)
        self.criterion = DETRLoss(matcher=matcher, eos_coef=args.eos_coef, cost_is_cluster=args.cost_is_cluster, cost_is_mention=args.cost_is_mention,
                                cost_coref=args.cost_coref, args=args)
        self.criterion.to(device)

        self.num_queries = args.num_queries
        hidden_dim = self.transformer.d_model
        self.query_embed = nn.Embedding(args.num_queries, hidden_dim)
        self.is_cluster = nn.Linear(hidden_dim, 1)
        self.args = args
        if args.single_distribution_queries:
            self.query_mu = nn.Parameter(torch.randn(1, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(1, hidden_dim))
        else:
            self.query_mu = nn.Parameter(torch.randn(args.num_queries, hidden_dim))
            self.query_sigma = nn.Parameter(torch.randn(args.num_queries, hidden_dim))

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

        self.cluster_threshold = self.args.threshold if self.args.threshold > 0 else 0.5
        self.coref_threshold = self.args.threshold if self.args.threshold > 0 else 0.5
        self.same_thresh_count = 0
        self.thresh_delta = 0.2
        self.train_evaluator = CorefEvaluator()
        self.eval_evaluator = CorefEvaluator()

        self.input_ids_pads = torch.ones(1, self.args.max_seq_length, dtype=torch.int, device=self.args.device) * TOKENS_PAD
        self.mask_pads = torch.zeros(1, self.args.max_seq_length, dtype=torch.int, device=self.args.device)
        self.recent_train_losses = []
        self.recent_train_losses_parts = {}
        self.losses = []
        self.losses_parts = {}
        self.batch_sizes = []
        self.all_cluster_logits = []
        self.all_coref_logits = []
        self.all_mentions = []
        self.all_predicted_clusters = []
        self.all_input_ids = []
        self.all_gold_clusters = []
        self.best_f1 = 0
        self.best_f1_epoch = -1
        self.last_saved_epoch = -1
        self.epoch = 0
        self.step_num = 0

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

            longformer_emb = self.longformer(masked_ids, attention_mask=masked_mask)[0]
            longfomer_no_pad_list.append(longformer_emb.reshape(-1, longformer_emb.shape[-1]))

        if self.args.random_queries:
            raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 1) * self.query_sigma + self.query_mu #raw_query_embed = torch.normal(torch.zeros_like(self.query_embed.weight), 0.5)
        else:
            raw_query_embed = self.query_embed.weight

        if not self.args.use_gold_mentions:
            hs, memory = self.transformer(self.input_proj(longfomer_no_pad_list), mask, raw_query_embed) # [dec_layers, 1, num_queries, emb], [1, seg*seq, emb]
        else:
            span_starts = [torch.tensor([m[0] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_ends = [torch.tensor([m[1] for m in gold_mentions[i]], dtype=torch.long) for i in range(len(gold_mentions))]
            span_emb, span_mask = self.get_span_emb(longfomer_no_pad_list, span_starts, span_ends, num_mentions)  # [mentions, emb']
            span_emb = self.span_proj(span_emb) # [mentions, emb]
            hs, memory = self.transformer(span_emb, span_mask, raw_query_embed)  # [dec_layers, bs, num_queries, emb], [bs, mentions, emb]


        last_hs = hs[-1] # [1, num_queries, emb]
        cluster_logits, coref_logits, mention_logits = self.calc_cluster_and_coref_logits(last_hs, memory, span_mask, gold_mentions.shape[1])

        out = {"coref_logits": coref_logits,
                "cluster_logits": cluster_logits,
                "mention_logits": mention_logits}
        return out

    def calc_cluster_and_coref_logits(self, last_hs, memory, span_mask, max_num_mentions):
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

    def get_span_emb(self, context_outputs_list, span_starts, span_ends, num_mentions):
        max_mentions = num_mentions.max()
        span_mask_list = []
        span_emb_list = []
        for i in range(len(num_mentions)):
            span_emb_construct = []
            span_start_emb = context_outputs_list[i][span_starts[i][:num_mentions[i]]] # [k, emb]
            span_emb_construct.append(span_start_emb)

            span_end_emb = context_outputs_list[i][span_ends[i][:num_mentions[i]]]  # [k, emb]
            span_emb_construct.append(span_end_emb)

            span_width = (1 + span_ends[i][:num_mentions[i]] - span_starts[i][:num_mentions[i]]).clamp(max=30)  # [k]

            span_width_index = span_width - 1  # [k]
            span_width_emb = self.span_width_embed.weight[span_width_index]
            # TODO add dropout
            # span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_construct.append(span_width_emb)

            mention_word_score = self.get_masked_mention_word_scores(context_outputs_list[i], span_starts[i][:num_mentions[i]], span_ends[i][:num_mentions[i]])  # [K, T]
            head_attn_reps = torch.matmul(mention_word_score, context_outputs_list[i])  # [K, emb]
            span_emb_construct.append(head_attn_reps)
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
                gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device)
            else:
                gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=self.args.device) for gm in gold_mentions_list]
            input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = \
                tensor_and_remove_empty(batch, gold_mentions_list, self.args, self.input_ids_pads, self.mask_pads)

            if len(input_ids) == 0:  #TODO fix?
                print(f'skipped {batch_idx}')
                return
            
            gold_matrix = create_gold_matrix(self.args.device, sum_text_len, self.args.num_queries, gold_clusters, gold_mentions_list)

            outputs = self(input_ids, input_mask, gold_mentions, num_mentions)
            cluster_logits, coref_logits, mention_logits = \
                outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

            if len(mention_logits) > 0:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
                                                            self.coref_threshold, self.cluster_threshold, gold_mentions_list)
            else:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                            self.coref_threshold, self.cluster_threshold, gold_mentions_list)
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
            gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, self.args.device)
        else:
            gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=self.args.device) for gm in gold_mentions_list]
        input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = tensor_and_remove_empty(batch, gold_mentions_list, self.args, self.input_ids_pads, self.mask_pads)
        if len(input_ids) == 0:
            print(f'skipped {batch_idx}')
            return
            
        gold_matrix = create_gold_matrix(self.args.device, sum_text_len, self.args.num_queries, gold_clusters, gold_mentions_list)

        outputs = self(input_ids, input_mask, gold_mentions, num_mentions)
        cluster_logits, coref_logits, mention_logits = \
            outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

        loss, loss_parts = self.criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})
        self.losses.append(loss.mean().detach().cpu())
        for key in loss_parts.keys():
            if key in self.losses_parts.keys() and len(self.losses_parts[key]) > 0:
                self.losses_parts[key] += loss_parts[key]
            else:
                self.losses_parts[key] = loss_parts[key]
        self.batch_sizes.append(loss.shape[0]) 
        
        self.all_input_ids += input_ids    
        self.all_gold_clusters += gold_clusters
        self.all_cluster_logits += cluster_logits.detach().cpu()
        self.all_coref_logits += coref_logits.detach().cpu()
        self.all_mentions += gold_mentions_list

        # predicted_clusters = calc_predicted_clusters(cluster_logits.detach().cpu(), coref_logits.detach().cpu(), [], \
        #                 .5, gold_mentions_list, self.args.use_gold_mentions, self.args.use_topk_mentions, self.args.is_cluster, self.args.slots, self.args.min_cluster_size)
        return {'loss': loss}

    def eval_by_thresh(self, coref_threshold, cluster_threshold):
        evaluator = CorefEvaluator()
        all_predicted_clusters = []
        metrics = [0] * 5    
        for i, (cluster_logits, coref_logits, gold_clusters, gold_mentions) in enumerate(
                zip(self.all_cluster_logits, self.all_coref_logits, self.all_gold_clusters, self.all_mentions)): 
            predicted_clusters = calc_predicted_clusters(cluster_logits.unsqueeze(0), coref_logits.unsqueeze(0), [], \
                coref_threshold, cluster_threshold, [gold_mentions])
            all_predicted_clusters += predicted_clusters
            evaluator.update(predicted_clusters, [gold_clusters])
        p, r, f1 = evaluator.get_prf()
        return p, r, f1, metrics, all_predicted_clusters

    def validation_epoch_end(self, val_step_outputs):
        eval_loss = np.average(self.losses, weights=self.batch_sizes)
        losses_parts = {key:np.average(self.losses_parts[key]) for key in self.losses_parts.keys()}

        if self.args.threshold > 0:
            p, r, f1, best_metrics, self.all_predicted_clusters = self.eval_by_thresh(self.coref_threshold, self.cluster_threshold)
        else:
            if self.thresh_delta == 0.2:
                thresh_coref_start = 0.05
                thresh_coref_end = 1
                thresh_cluster_start = 0.05
                thresh_cluster_end = 1
            else:
                thresh_coref_start = max(0.01, self.coref_threshold - 2*self.thresh_delta)
                thresh_coref_end = min(1, self.coref_threshold + 2.5*self.thresh_delta)
                thresh_cluster_start = max(0.01, self.cluster_threshold - 2*self.thresh_delta)
                thresh_cluster_end = min(1, self.cluster_threshold + 2.5*self.thresh_delta)
                
            best = [-1, -1, -1]
            best_metrics = []
            best_coref_threshold = None
            best_cluster_threshold = None
            for coref_threshold in tqdm(np.arange(thresh_coref_start, thresh_coref_end, self.thresh_delta), desc='Searching for best threshold'):
                for cluster_threshold in np.arange(thresh_cluster_start, thresh_cluster_end, self.thresh_delta):
                    p, r, f1, metrics, all_predicted_clusters = self.eval_by_thresh(coref_threshold, cluster_threshold)
                    if f1 > best[-1]:
                        best = p,r,f1
                        best_metrics = metrics
                        best_coref_threshold = coref_threshold
                        best_cluster_threshold = cluster_threshold
                        self.all_predicted_clusters = all_predicted_clusters
            p,r,f1 = best
            if best_cluster_threshold == self.cluster_threshold and best_coref_threshold == self.coref_threshold:
                self.same_thresh_count += 1
                if self.same_thresh_count == 5 and self.thresh_delta == 0.2:
                    self.thresh_delta = 0.02
                    self.same_thresh_count = 0
            else:
                self.same_thresh_count = 0
            self.cluster_threshold = best_cluster_threshold
            self.coref_threshold = best_coref_threshold

        print_predictions(self.all_predicted_clusters, self.all_gold_clusters, self.all_input_ids, self.args, self.tokenizer)
        prec_gold_to_one_pred, prec_pred_to_one_gold, avg_gold_split_without_perfect, avg_gold_split_with_perfect, \
            avg_pred_split_without_perfect, avg_pred_split_with_perfect, prec_biggest_gold_in_pred_without_perfect, \
                prec_biggest_gold_in_pred_with_perfect, prec_biggest_pred_in_gold_without_perfect, prec_biggest_pred_in_gold_with_perfect = \
                    error_analysis(self.all_predicted_clusters, self.all_gold_clusters)

        results = {'loss': eval_loss,
                'avg_f1': f1,
                'coref_threshold': self.coref_threshold,
                'cluster_threshold': self.cluster_threshold,
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
                save_checkpoint(self.args, self.epoch, self.coref_threshold, self.cluster_threshold, self, self.optimizer, output_dir)
                print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_epoch}')
                self.best_f1 = f1
                self.best_f1_epoch = self.epoch
                print(f'saved checkpoint with f1 {self.best_f1} in step {self.best_f1_epoch} to {output_dir}')
                if prev_best_f1_epoch > -1:
                    path_to_remove = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(prev_best_f1_epoch))
                    shutil.rmtree(path_to_remove)
                    print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')
            else:
                output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.epoch))
                save_checkpoint(self.args, self.epoch, self.coref_threshold, self.cluster_threshold, self, self.optimizer, output_dir)
                print(f'saved checkpoint in epoch {self.epoch}')
                path_to_remove = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.last_saved_epoch))
                if self.last_saved_epoch > -1 and self.last_saved_epoch != self.best_f1_epoch and os.path.exists(path_to_remove):
                    shutil.rmtree(path_to_remove)
                    print(f'removed previous checkpoint in epoch {self.last_saved_epoch}')
                self.last_saved_epoch = self.epoch
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
        self.all_mentions = []
        self.all_input_ids = []
        self.all_gold_clusters = []
        self.all_predicted_clusters = []
        self.query_cluster_confusion_matrix = np.zeros([self.args.num_queries, self.args.num_queries], dtype=int)
        # self.query_cluster_confusion_matrix = np.zeros([self.args.num_queries, 26], dtype=int)

class DETRLoss(nn.Module):
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
        matched_predicted_cluster_id, matched_gold_cluster_id = self.matcher(outputs, targets)

        targets_clusters = targets['clusters']
        targets_mentions = targets['mentions']
        bs = outputs["coref_logits"].shape[0]
        costs = []
        costs_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[]}
        for i in range(bs):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            coref_logits = outputs["coref_logits"][i].squeeze(0)  # [num_queries, tokens]
            cluster_logits = outputs["cluster_logits"][i].squeeze() # [num_queries]
            if self.args.add_junk:
                mention_logits = outputs["mention_logits"][i].squeeze() # [tokens]
            num_queries, doc_len = coref_logits.shape
            #TODO: normalize according to number of clusters? (identical to DETR)

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
                cost_coref = F.binary_cross_entropy(permuted_coref_logits, permuted_gold, reduction='mean')
            elif coref_logits.shape[1] > 0:
                cost_coref = F.binary_cross_entropy(coref_logits, torch.zeros_like(coref_logits), reduction='mean')

            costs_parts['loss_is_cluster'].append(self.cost_is_cluster * cost_is_cluster.detach().cpu())
            costs_parts['loss_is_mention'].append(self.cost_is_mention * cost_is_mention.detach().cpu())
            costs_parts['loss_coref'].append(self.cost_coref * cost_coref.detach().cpu())
            total_cost = self.cost_coref * cost_coref + self.cost_is_cluster * cost_is_cluster + self.cost_is_mention * cost_is_mention
            costs.append(total_cost)
        return torch.stack(costs), costs_parts
