import os
import logging
import random
import numpy as np
# import wandb
import torch
from eval import evaluate, report_eval
from tqdm import tqdm, trange
import time, datetime
from misc import save_on_master, is_main_process
from utils import tensor_and_remove_empty, create_gold_matrix, calc_predicted_clusters, create_junk_gold_mentions, try_measure_len
from optimization import WarmupLinearSchedule, WarmupExponentialSchedule
import itertools
from metrics import CorefEvaluator
from utils import load_from_checkpoint, save_checkpoint
from consts import TOKENS_PAD, SPEAKER_PAD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from detr import DETRDataModule, DETR

from transformers import AdamW, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    epoch_iterator, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args, evaluator, skip_steps, recent_grad_norms, recent_cluster_logits,
        recent_coref_logits, recent_losses, recent_losses_parts, recent_logits_sums, global_step, lr_scheduler, eval_loader, eval_dataset, threshold):
    for step, batch in enumerate(epoch_iterator):
        if batch == []:
            continue
        if skip_steps > 0:
            skip_steps -= 1
            continue

        model.train()


        # recent_cluster_logits.append(cluster_logits.detach().cpu().numpy())
        # recent_coref_logits.append(coref_logits.detach().cpu().numpy().flatten())
        # recent_logits_sums.append(coref_logits.detach().sum(1).flatten().cpu().numpy())

        # TODO handle NaNs and +-infs

        if args.n_gpu > 1 or args.train_batch_size > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # if args.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        #     total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        # else:
        # print("before")
        # print(model.IO_score[0].weight)
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # recent_grad_norms.append(total_norm.item())
        # epoch_iterator.set_postfix({'loss': loss.item()})

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if args.lr_drop_interval == 'step':
                lr_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += args.train_batch_size

        # print("after")
        # print(model.IO_score[0].weight)
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    return global_step, threshold


def train(args, model, wandb=None):
    """ Train the model """
    # output_dir = Path(args.output_dir)

    logger.info("Training/evaluation parameters %s", args)
    # if args.resume_from:
    #     model = DETR.load_from_checkpoint(args.resume_from)
    data_model = DETRDataModule(args)
    if wandb is not None:        
        trainer = pl.Trainer(max_epochs=args.num_train_epochs, gpus=args.n_gpu, amp_backend='apex', logger= wandb, accumulate_grad_batches=args.gradient_accumulation_steps,\
            callbacks=[ModelCheckpoint(monitor="eval_avg_f1"), ModelCheckpoint(monitor="epoch")], default_root_dir=args.output_dir)
    else:
        trainer = pl.Trainer(max_epochs=args.num_train_epochs, gpus=args.n_gpu, amp_backend='apex', accumulate_grad_batches=args.gradient_accumulation_steps,\
            callbacks=[ModelCheckpoint(monitor="eval_avg_f1"), ModelCheckpoint(monitor="epoch")], default_root_dir=args.output_dir)
                             
    # global_step = 0 if not args.resume_from else args.resume_global_step
    # if args.local_rank in [-1, 0]:
    #     purge_step = None if not args.resume_from else args.resume_global_step

    # # Train!
    # logger.info("***** Running training *****")
    # # logger.info("  Num steps per epoch = %d", try_measure_len(train_loader))
    # logger.info("  Num Epochs = %d", args.num_train_epochs if args.num_train_epochs is not None else -1)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # # logger.info("  Total optimization steps = %d", args.t_total)


    # # model.zero_grad()

    # train_iterator = itertools.count() if args.num_train_epochs is None else range(int(args.num_train_epochs))
    # train_iterator = tqdm(train_iterator, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        print(args.output_dir)
        os.makedirs(args.output_dir)

    trainer.fit(model, data_model)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
