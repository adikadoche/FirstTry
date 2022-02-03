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
from metrics import CorefEvaluator, MentionEvaluator
from utils import create_target_and_predict_matrix, load_from_checkpoint, save_checkpoint
from coref_analysis import print_predictions

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    epoch_iterator, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args, skip_steps, recent_losses, recent_losses_parts, global_step, lr_scheduler, steps_from_last_log):
    for step, batch in enumerate(epoch_iterator):
        if skip_steps > 0:
            skip_steps -= 1
            continue

        model.train()

        gold_clusters_list = batch[-1]
        batch = tuple(tensor.to(args.device) for tensor in batch[:-1])
        input_ids, input_mask, gold_mentions, gold_mentions_mask = batch

        sum_text_len = torch.sum(input_mask, -1)

        gold_mentions_list = [[list(set([tuple(m) for c in gc for m in c])) for gc in x] for x in gold_clusters_list]

        gold_matrix = create_gold_matrix(args.device, sum_text_len, args.num_queries, gold_clusters_list, gold_mentions_list, gold_mentions.shape[2])
        max_mentions = torch.tensor(gold_mentions.shape[1], device=gold_mentions.device) if args.use_gold_mentions \
            else torch.max(sum_text_len) // int(-10*args.topk_lambda +6)
        max_mentions = max_mentions.repeat([input_ids.shape[0], 1])

        outputs = model(input_ids, max_mentions, input_mask, gold_mentions, gold_mentions_mask)
        mentions_list = outputs['mentions']
        mentions_list = mentions_list.detach().cpu().numpy()
        mentions_list = [[(m[0], m[1]) for m in mentions_list[j] if m[0] != -1 and m[1] != -1] for j in range(mentions_list.shape[0])]

        gold_matrix = gold_matrix.reshape(gold_matrix.shape[0]*gold_matrix.shape[1], gold_matrix.shape[2], gold_matrix.shape[3])
        gold_mentions_list = [ml for gml in gold_mentions_list for ml in gml]
        gold_matrix, outputs['coref_logits'], dist_matrix, goldgold_dist_mask, junkgold_dist_mask = \
            create_target_and_predict_matrix( \
            gold_mentions_list, mentions_list, gold_matrix, outputs['coref_logits'], outputs['inputs'])

        loss, loss_parts = criterion(outputs, {'clusters':gold_matrix},
            dist_matrix, goldgold_dist_mask, junkgold_dist_mask)

        if loss.numel() > 1:
            loss = loss.mean() * (torch.sum(sum_text_len) / args.max_total_seq_len)  # mean() to average on multi-gpu parallel training   ####
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        recent_losses.append(loss.item())
        for key in loss_parts.keys():
            if key in recent_losses_parts.keys() and len(recent_losses_parts[key]) > 0:
                recent_losses_parts[key] += loss_parts[key]
            else:
                recent_losses_parts[key] = loss_parts[key]
        epoch_iterator.set_postfix({'loss': loss.item()})
        global_step += input_ids.shape[0] * input_ids.shape[1]
        steps_from_last_log += input_ids.shape[0] * input_ids.shape[1]

        if (step + 1) % args.gradient_accumulation_steps == 0:
            for _ in range(input_ids.shape[0] * input_ids.shape[1]):
                optimizer.step()
            if args.lr_drop_interval == 'step':
                for _ in range(input_ids.shape[0] * input_ids.shape[1]):
                    lr_scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and steps_from_last_log > args.logging_steps:   ##
            if not args.is_debug:
                dict_to_log = {}
                dict_to_log['lr'] = optimizer.param_groups[2]['lr']
                dict_to_log['lr_bert'] = optimizer.param_groups[0]['lr']
                dict_to_log['loss'] = np.mean(recent_losses)
                for key in recent_losses_parts.keys():
                    dict_to_log[key] = np.mean(recent_losses_parts[key])
                wandb.log(dict_to_log, step=global_step)
            recent_losses.clear()
            recent_losses_parts.clear()
            steps_from_last_log = 0
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    return global_step, steps_from_last_log

def create_optimization(model, args, train_loader):
    no_decay = ['bias', 'LayerNorm.weight']
    back_params = [args.backbone_name]

    model_decay = [p for n, p in model.named_parameters() if
                   any(hp in n for hp in back_params) and not any(nd in n for nd in no_decay)]
    model_no_decay = [p for n, p in model.named_parameters() if
                      any(hp in n for hp in back_params) and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in model.named_parameters() if
                  not any(hp in n for hp in back_params) and not any(nd in n for nd in no_decay)]
    head_no_decay = [p for n, p in model.named_parameters() if
                     not any(hp in n for hp in back_params) and any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': model_decay, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': model_no_decay, 'lr': args.lr_backbone, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': head_no_decay, 'lr': args.lr, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr) #TODO check betas and epsilon
                    #   betas=(args.adam_beta1, args.adam_beta2),
                    #   eps=args.adam_epsilon)

    if args.max_steps > 0:
        args.t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        args.t_total = train_loader.num_examples() // args.gradient_accumulation_steps * args.num_train_epochs

    # lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps / args.train_batch_size))
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.t_total)
    # lr_scheduler = WarmupExponentialSchedule(optimizer, warmup_steps=int(args.warmup_steps / args.train_batch_size),
    #                                     gamma=0.99998)  # ConstantLRSchedule(optimizer)

    return optimizer, lr_scheduler

def train(args, model, criterion, train_loader, eval_loader, eval_dataset):
    """ Train the model """
    # output_dir = Path(args.output_dir)

    logger.info("Training/evaluation parameters %s", args)
    optimizer, lr_scheduler = create_optimization(model, args, train_loader)

    thresh_delta = 0.2    
    coref_threshold, cluster_threshold = 0.5, 0.5 # starting threshold, later fixed by eval
    best_f1 = -1
    best_f1_global_step = -1
    last_saved_global_step = -1
    if args.resume_from:
        logger.info("Loading from checkpoint {}".format(args.resume_from))
        loaded_args = load_from_checkpoint(model, args.resume_from, args.device, optimizer, lr_scheduler)
        coref_threshold = loaded_args['numbers']['coref_threshold']
        cluster_threshold = loaded_args['numbers']['cluster_threshold']
        best_f1_global_step = loaded_args['numbers']['best_f1_global_step']
        last_saved_global_step = loaded_args['numbers']['last_saved_global_step']
        best_f1 = loaded_args['numbers']['best_f1']
        thresh_delta = loaded_args['numbers']['thresh_delta']   

        if args.reset_optim:
            optimizer, lr_scheduler = create_optimization(model, args, train_loader)
            args.resume_global_step = 0
        else:
            args.resume_global_step = int(loaded_args['global_step'])
            args.num_train_epochs = (args.t_total - args.resume_global_step) * args.gradient_accumulation_steps // train_loader.num_examples()
        
            results = report_eval(args, eval_loader, eval_dataset, args.resume_global_step, model, criterion, coref_threshold, cluster_threshold, thresh_delta)

        if not args.do_train:
            return args.resume_global_step

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

    global_step = 0 if not args.resume_from else args.resume_global_step
    if args.local_rank in [-1, 0]:
        purge_step = None if not args.resume_from else args.resume_global_step

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"  Number of non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    for key in ['span', 'score', 'backbone', 'slot', 'longformer']:
        n_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and key in n)
        if n_train > 0:
            logger.info(f"    {key} trainable parameters: {n_train:,}")
        n_notrain = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad and key in n)
        if n_notrain > 0:
            logger.info(f"    {key} non-trainable parameters: {n_notrain:,}")
    logger.info("  Num steps per epoch = %d", try_measure_len(train_loader))
    logger.info("  Num Epochs = %d", args.num_train_epochs if args.num_train_epochs is not None else -1)
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
    steps_from_last_log = 0

    start_time = time.time()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration in Epoch {}".format(epoch), disable=args.local_rank not in [-1, 0], leave=False)
        global_step, steps_from_last_log = train_one_epoch(   
            model, criterion, epoch_iterator, optimizer, scaler, args, skip_steps, recent_losses, recent_losses_parts, global_step,
            lr_scheduler, steps_from_last_log)

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

            cluster_evaluator, mention_evaluator, men_propos_evaluator = eval_train(train_loader, eval_dataset, args, model, cluster_threshold, coref_threshold)

            p_train, r_train, f1_train = cluster_evaluator.get_prf()
            pm_train, rm_train, f1m_train = mention_evaluator.get_prf()
            pmp_train, rmp_train, f1mp_train = men_propos_evaluator.get_prf()
            if args.local_rank in [-1, 0]:
                if not args.is_debug:
                    dict_to_log = {}
                    dict_to_log['Train Precision'] = p_train
                    dict_to_log['Train Recall'] = r_train
                    dict_to_log['Train F1'] = f1_train
                    dict_to_log['Train Mention Precision'] = pm_train
                    dict_to_log['Train Mention Recall'] = rm_train
                    dict_to_log['Train Mention F1'] = f1m_train
                    dict_to_log['Train MentionProposal Precision'] = pmp_train
                    dict_to_log['Train MentionProposal Recall'] = rmp_train
                    dict_to_log['Train MentionProposal F1'] = f1mp_train
                    wandb.log(dict_to_log, step=global_step)
                logger.info('Train f1, precision, recall: {}, Mentions f1, precision, recall: {}, Mention Proposals f1, precision, recall: {}'.format(\
                    (f1_train, p_train, r_train), (f1m_train, pm_train, rm_train), (f1mp_train, pmp_train, rmp_train)))

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


def eval_train(train_dataloader, eval_dataset, args, model, cluster_threshold, coref_threshold):    
    all_cluster_logits_cuda = []
    all_coref_logits_cuda = []
    all_mention_logits_cuda = []
    all_gold_clusters = []
    all_input_ids = []
    all_gold_mentions = []

    cluster_train_evaluator = CorefEvaluator()
    mention_train_evaluator = MentionEvaluator()
    men_propos_train_evaluator = MentionEvaluator()
    for batch in tqdm(train_dataloader, desc="Evaluating Train"):
        model.eval()
        gold_clusters_list = batch[-1]
        batch = tuple(tensor.to(args.device) for tensor in batch[:-1])
        input_ids, input_mask, gold_mentions, gold_mentions_mask = batch

        sum_text_len = torch.sum(input_mask, -1)

        max_mentions = torch.tensor(gold_mentions.shape[1], device=gold_mentions.device) if args.use_gold_mentions \
            else torch.max(sum_text_len) // int(-10*args.topk_lambda +6)
        max_mentions = max_mentions.repeat([input_ids.shape[0], 1])

        with torch.no_grad():
            outputs = model(input_ids, max_mentions, input_mask, gold_mentions, gold_mentions_mask)
            cluster_logits, coref_logits, mentions_list = \
                outputs['cluster_logits'], outputs['coref_logits'].clone(), outputs['mentions'].detach().cpu().numpy()
            mentions_list = [[(m[0], m[1]) for m in mentions_list[j] if m[0] != -1 and m[1] != -1] for j in range(mentions_list.shape[0])]

            gold_clusters_list = [gc for g in gold_clusters_list for gc in g]
            input_ids = input_ids.reshape(input_ids.shape[0]*input_ids.shape[1], -1)

            for i in range(cluster_logits.shape[0]):
                predicted_clusters = calc_predicted_clusters(cluster_logits[i].unsqueeze(0).cpu().detach(), \
                    coref_logits[i].unsqueeze(0).cpu().detach(), [], coref_threshold, cluster_threshold, \
                        [mentions_list[i]], args.slots)
                cluster_train_evaluator.update(predicted_clusters, [gold_clusters_list[i]])
                gold_mentions_e = [[[]]] if [gold_clusters_list[i]] == [[]] or [gold_clusters_list[i]] == [()] \
                    else [[[m for d in c for m in d]] for c in [gold_clusters_list[i]]]
                predicted_mentions_e = [[[]]] if predicted_clusters == [[]] or predicted_clusters == [()] \
                    else [[[m for d in c for m in d]] for c in predicted_clusters]
                mention_train_evaluator.update(predicted_mentions_e, gold_mentions_e)
                men_propos_train_evaluator.update([[mentions_list[i]]], gold_mentions_e)

                all_gold_mentions += [mentions_list[i]]
                all_input_ids += input_ids[i].unsqueeze(0)   
                all_gold_clusters += [gold_clusters_list[i]]
            
        all_cluster_logits_cuda += [cl.detach().clone() for cl in cluster_logits]
        all_coref_logits_cuda += [cl.detach().clone() for cl in coref_logits]

    print("============ TRAIN EXAMPLES ============")
    print_predictions(all_cluster_logits_cuda, all_coref_logits_cuda, all_mention_logits_cuda, \
        all_gold_clusters, all_gold_mentions, all_input_ids, coref_threshold, cluster_threshold, args, \
            eval_dataset.tokenizer)

    return cluster_train_evaluator, mention_train_evaluator, men_propos_train_evaluator