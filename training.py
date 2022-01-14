import shutil
import os
import logging
import random
import numpy as np
import wandb
import torch
from eval import evaluate, report_eval
from tqdm import tqdm, trange
import time, datetime
from misc import save_on_master, is_main_process
from utils import tensor_and_remove_empty, create_gold_matrix, calc_predicted_clusters, create_junk_gold_mentions, try_measure_len
from optimization import WarmupLinearSchedule, WarmupExponentialSchedule
import itertools
from metrics import CorefEvaluator
from utils import load_from_checkpoint, save_checkpoint, create_target_and_predict_matrix

from transformers import AdamW, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    epoch_iterator, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args, cluster_evaluator, mention_evaluator, skip_steps, recent_losses, recent_losses_parts, global_step, lr_scheduler,
        coref_threshold, cluster_threshold):
    for step, batch in enumerate(epoch_iterator):
        if skip_steps > 0:
            skip_steps -= 1
            continue

        model.train()

        sum_text_len = [sum(tl) for tl in batch['text_len']]
        gold_clusters = batch['clusters']

        gold_mentions_list = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
        if args.add_junk:
            gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, args.device)
        else:
            gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=args.device) for gm in gold_mentions_list]

        input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = tensor_and_remove_empty(batch, gold_mentions_list, args)
        # if len(input_ids) == 0 or input_ids.shape[1] > 1:
        if len(input_ids) == 0:
            print(f"skipped {step}")
            continue

        gold_matrix = create_gold_matrix(args.device, sum_text_len, args.num_queries, gold_clusters, gold_mentions_list)
        max_mentions = torch.tensor(gold_mentions.shape[1], device=gold_mentions.device) if args.use_gold_mentions and not args.use_topk_mentions else sum_text_len.max()//2
        max_mentions = max_mentions.repeat([input_ids.shape[0], 1])

        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, max_mentions, input_mask, gold_mentions, gold_clusters, num_mentions)
                cluster_logits, coref_logits = outputs['cluster_logits'], outputs['coref_logits']

                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(),
                                                            coref_threshold, cluster_threshold, gold_mentions_list, args.slots)
                cluster_evaluator.update(predicted_clusters, gold_clusters)
                mention_evaluator.update([list(set([tuple(m) for c in gc for m in c])) for gc in predicted_clusters], gold_mentions_list)
                loss = criterion(outputs, gold_matrix)
        else:
            outputs = model(input_ids, max_mentions, input_mask, gold_mentions, gold_clusters, num_mentions)
            cluster_logits, coref_logits, mention_logits, mentions_list = \
                outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits'], outputs['mentions']
            mentions_list = mentions_list.detach().cpu().numpy()
            mentions_list = [[(m[0], m[1]) for m in mentions_list[j] if m[0] != -1 and m[1] != -1] for j in range(mentions_list.shape[0])]

            if args.use_topk_mentions:
                gold_matrix, gold_mentions_vector = create_target_and_predict_matrix(gold_mentions_list, mentions_list, gold_matrix)
            if args.add_junk:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
                                                            coref_threshold, cluster_threshold, mentions_list, args.slots)
            else:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                            coref_threshold, cluster_threshold, mentions_list, args.slots)
            cluster_evaluator.update(predicted_clusters, gold_clusters)
            gold_mentions_e = [[]] if gold_clusters == [[]] or gold_clusters == [()] else [[[m for c in gold_clusters for d in c for m in d]]]
            predicted_mentions_e = [[]] if predicted_clusters == [[]] or predicted_clusters == [()] else [[[m for c in predicted_clusters for d in c for m in d]]]
            mention_evaluator.update(predicted_mentions_e, gold_mentions_e)
            loss, loss_parts = criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})

        if args.n_gpu > 1 or args.train_batch_size > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # recent_grad_norms.append(total_norm.item())
        recent_losses.append(loss.item())
        for key in loss_parts.keys():
            if key in recent_losses_parts.keys() and len(recent_losses_parts[key]) > 0:
                recent_losses_parts[key] += loss_parts[key]
            else:
                recent_losses_parts[key] = loss_parts[key]
        epoch_iterator.set_postfix({'loss': loss.item()})

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

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if not args.is_debug:
                    dict_to_log = {}
                    dict_to_log['lr'] = optimizer.param_groups[0]['lr']
                    dict_to_log['lr_bert'] = optimizer.param_groups[1]['lr']
                    dict_to_log['loss'] = np.mean(recent_losses)
                    for key in recent_losses_parts.keys():
                        dict_to_log[key] = np.mean(recent_losses_parts[key])
                    wandb.log(dict_to_log, step=global_step)
                recent_losses.clear()
                recent_losses_parts.clear()
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    return global_step, coref_threshold, cluster_threshold


def train(args, model, criterion, train_loader, eval_loader, eval_dataset):
    """ Train the model """
    # output_dir = Path(args.output_dir)

    logger.info("Training/evaluation parameters %s", args)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if args.backbone_name not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if args.backbone_name in n and p.requires_grad],
            "lr": args.lr_backbone, #TODO: learn how to freeze backbone
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    if args.max_steps > 0:
        args.t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        args.t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps / args.train_batch_size))
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps // args.train_batch_size,
                                        t_total=args.t_total)  # ConstantLRSchedule(optimizer)
    # lr_scheduler = WarmupExponentialSchedule(optimizer, warmup_steps=int(args.warmup_steps / args.train_batch_size),
    #                                     gamma=0.99998)  # ConstantLRSchedule(optimizer)

    thresh_delta = 0.2    
    coref_threshold, cluster_threshold = 0.5, 0.5 # starting threshold, later fixed by eval
    best_f1 = -1
    best_f1_global_step = -1
    last_saved_global_step = -1
    if args.resume_from:
        logger.info("Loading from checkpoint {}".format(args.resume_from))
        loaded_args = load_from_checkpoint(model, args.resume_from, args.device, optimizer, lr_scheduler)
        args.resume_global_step = int(loaded_args['global_step'])
        coref_threshold = loaded_args['numbers']['coref_threshold']
        cluster_threshold = loaded_args['numbers']['cluster_threshold']
        best_f1_global_step = loaded_args['numbers']['best_f1_global_step']
        last_saved_global_step = loaded_args['numbers']['last_saved_global_step']
        best_f1 = loaded_args['numbers']['best_f1']
        thresh_delta = loaded_args['numbers']['thresh_delta']   
        if not args.do_train:
            return args.resume_global_step
        args.num_train_epochs = (args.t_total - args.resume_global_step) * args.gradient_accumulation_steps // len(train_loader)

    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print("FML")

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    
    
    if args.train_batch_size > 1:
        args.eval_steps = -1 if args.eval_steps == -1 else max(1, int(round(args.eval_steps / args.train_batch_size)))
        args.save_steps = -1 if args.save_steps == -1 else max(1, int(round(args.save_steps / args.train_batch_size)))
        args.logging_steps = -1 if args.logging_steps == -1 else max(1, int(round(args.logging_steps / args.train_batch_size)))


    global_step = 0 if not args.resume_from else args.resume_global_step
    if args.local_rank in [-1, 0]:
        purge_step = None if not args.resume_from else args.resume_global_step

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num steps per epoch = %d", try_measure_len(train_loader))
    logger.info("  Num Epochs = %d", args.num_train_epochs if args.num_train_epochs is not None else -1)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)


    model.zero_grad()
    recent_losses = []
    recent_losses_parts = {}
    train_iterator = itertools.count() if args.num_train_epochs is None else range(int(args.num_train_epochs))
    train_iterator = tqdm(train_iterator, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    skip_steps = args.skip_steps
    same_thresh_count = 0

    start_time = time.time()
    for epoch in train_iterator:
        # if epoch > len(train_iterator) / 2:
        #     args.add_junk = True
        cluster_evaluator = CorefEvaluator()
        mention_evaluator = CorefEvaluator()
        epoch_iterator = tqdm(train_loader, desc="Iteration in Epoch {}".format(epoch), disable=args.local_rank not in [-1, 0], leave=False)
        global_step, coref_threshold, cluster_threshold = train_one_epoch(   
            model, criterion, epoch_iterator, optimizer, scaler, args, cluster_evaluator, mention_evaluator, skip_steps, recent_losses, recent_losses_parts, global_step,
            lr_scheduler, coref_threshold, cluster_threshold)

        p_train, r_train, f1_train = cluster_evaluator.get_prf()
        pm_train, rm_train, f1m_train = mention_evaluator.get_prf()
        if args.local_rank in [-1, 0]:
            if not args.is_debug:
                dict_to_log = {}
                dict_to_log['Train Precision'] = p_train
                dict_to_log['Train Recall'] = r_train
                dict_to_log['Train F1'] = f1_train
                dict_to_log['Train Mention Precision'] = pm_train
                dict_to_log['Train Mention Recall'] = rm_train
                dict_to_log['Train Mention F1'] = f1m_train
                wandb.log(dict_to_log, step=global_step)
            logger.info('Train f1, precision, recall: {}, Mentions f1, precision, recall: {}'.format((f1_train, p_train, r_train), (f1m_train, pm_train, rm_train)))

        if args.lr_drop_interval == 'epoch':
            lr_scheduler.step()  # Update learning rate schedule

        if args.local_rank in [-1, 0]:
            if args.eval_epochs > 0 and (epoch + 1) % args.eval_epochs == 0 or \
                    epoch + 1 == args.num_train_epochs and (args.eval_epochs > 0 or args.eval_steps > 0):
                results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, coref_threshold, cluster_threshold, thresh_delta)
                new_coref_threshold, new_cluster_threshold, eval_f1 = results['coref_threshold'], results['cluster_threshold'], results['avg_f1']

                if new_cluster_threshold == cluster_threshold and new_coref_threshold == coref_threshold:
                    same_thresh_count += 1
                    if same_thresh_count == 5 and thresh_delta == 0.2:
                        thresh_delta = 0.02
                        same_thresh_count = 0
                else:
                    same_thresh_count = 0
                cluster_threshold = new_cluster_threshold
                coref_threshold = new_coref_threshold

            if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0 or epoch + 1 == args.num_train_epochs:
                if eval_f1 > best_f1:
                    numbers = {'coref_threshold':coref_threshold, 
                        'cluster_threshold': cluster_threshold, 
                        'thresh_delta': thresh_delta,
                        'best_f1_global_step': global_step,
                        'last_saved_global_step': last_saved_global_step,
                        'best_f1': eval_f1}
                    prev_best_f1 = best_f1
                    prev_best_f1_global_step = best_f1_global_step
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    save_checkpoint(args, global_step, numbers, model, optimizer, lr_scheduler, output_dir)
                    print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_global_step}')
                    best_f1 = eval_f1
                    best_f1_global_step = global_step
                    print(f'saved checkpoint with f1 {best_f1} in step {best_f1_global_step} to {output_dir}')
                    path_to_remove = os.path.join(args.output_dir, 'checkpoint-{}'.format(prev_best_f1_global_step))
                    if prev_best_f1_global_step > -1 and os.path.exists(path_to_remove):
                        shutil.rmtree(path_to_remove)
                        print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')
                else:
                    numbers = {'coref_threshold':coref_threshold, 
                        'cluster_threshold': cluster_threshold, 
                        'thresh_delta': thresh_delta,
                        'best_f1_global_step': best_f1_global_step,
                        'last_saved_global_step': global_step,
                        'best_f1': best_f1}
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    save_checkpoint(args, global_step, numbers, model, optimizer, lr_scheduler, output_dir)
                    print(f'saved checkpoint in global_step {global_step}')
                    path_to_remove = os.path.join(args.output_dir, 'checkpoint-{}'.format(last_saved_global_step))
                    if last_saved_global_step > -1 and last_saved_global_step != best_f1_global_step and os.path.exists(path_to_remove):
                        shutil.rmtree(path_to_remove)
                        print(f'removed previous checkpoint in global_step {last_saved_global_step}')
                    last_saved_global_step = global_step
                if not args.is_debug:
                    wandb.log({'eval_best_f1':best_f1}, step=global_step)
                    try:
                        wandb.log({'eval_best_f1_checkpoint':os.path.join(args.output_dir, 'checkpoint-{}'.format(best_f1_global_step))}, step=global_step)
                    except:
                        pass


        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return global_step

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
