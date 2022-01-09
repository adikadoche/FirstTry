import json
import os
import glob
import logging
from collections import Counter
import random
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
from tqdm import tqdm
from coref_bucket_batch_sampler import BucketBatchSampler
from coref_analysis import print_predictions, error_analysis
from data import get_dataset
from utils import tensor_and_remove_empty, calc_best_avg_f1, create_gold_matrix, try_measure_len, load_from_checkpoint, create_junk_gold_mentions
from conll import evaluate_conll
from consts import TOKENS_PAD, SPEAKER_PAD
import wandb
logger = logging.getLogger(__name__)



def report_eval(args, eval_dataloader, eval_dataset, global_step, model, criterion, coref_threshold, cluster_threshold, thresh_delta=0.02):
    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(args, eval_dataloader, eval_dataset, model, criterion, str(global_step), coref_threshold, cluster_threshold, thresh_delta)
        if not args.is_debug:
            dict_to_log = {}
            for key, value in results.items():
                dict_to_log['eval_{}'.format(key)] = value
            wandb.log(dict_to_log, step=global_step)
        return results
    return None

def make_evaluation(model, criterion, eval_loader, eval_dataset, args):  
    # Evaluation 'no', 'specific', 'all', 'vanilla'
    if args.eval == 'specific':
        checkpoint = args.output_dir
        loaded_args = load_from_checkpoint(model, checkpoint, args.device)
        global_step = loaded_args['global_step']
        evaluate(args, eval_loader, model, criterion, global_step)
    elif args.eval == 'vanilla':
        evaluate(args, eval_loader, model, criterion, '0')
    elif args.eval == 'all':
        import time

        original_output_dir = args.output_dir.rstrip('/')
        args.output_dir = args.output_dir.rstrip('/') + '_eval'
        logger.info("Evaluation output is: %s", args.output_dir)

        evaluated = set()
        best_checkpoint = ''
        best_f1 = 0
        second_best_f1 = 0
        second_best_checkpoint = ''
        while True:
            if args.resume_from and not args.do_train:
                checkpoints_file = list(
                    os.path.dirname(c) for c in glob.glob(original_output_dir + '/model*.pt', recursive=True))
                checkpoints_dir = list(
                    os.path.dirname(c) for c in glob.glob(original_output_dir + '/**/model*.pt', recursive=True))
                checkpoints = set(checkpoints_dir+checkpoints_file)
            else:
                checkpoints = list(
                    os.path.dirname(c) for c in glob.glob(original_output_dir + '/**/model*.pt', recursive=True))
            checkpoints = [c for c in checkpoints if c not in evaluated]
            if args.eval_skip_until > 0:
                checkpoints = [c for c in checkpoints if int(c.split('-')[-1]) >= args.eval_skip_until]

            if len(checkpoints) > 0:
                logger.info("Evaluating the following checkpoints: %s", checkpoints)

                try:
                    for checkpoint in checkpoints:
                        loaded_args = load_from_checkpoint(model, args.resume_from, args.device)
                        global_step = int(loaded_args['global_step'])
                        coref_threshold = loaded_args['numbers']['coref_threshold']
                        cluster_threshold = loaded_args['numbers']['cluster_threshold']
                        thresh_delta = loaded_args['numbers']['thresh_delta']
                        results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, coref_threshold, cluster_threshold, thresh_delta)
                        if results['avg_f1'] > best_f1:
                            best_checkpoint = checkpoint
                            best_f1 = results['avg_f1']
                        elif second_best_f1 < results['avg_f1'] < best_f1:
                            second_best_f1 = results['avg_f1']
                            second_best_checkpoint = checkpoint

                    evaluated.update(checkpoints)
                except Exception as e:
                    logger.error(
                        "Got an exception. Will sleep for {} and try again. {}".format(args.eval_sleep, repr(e)))
                    time.sleep(args.eval_sleep)
            else:
                logger.info("No new checkpoints. Best F1 is {} in checkpoint {}. Second best F1 is {} in checkpoint {}. Sleep for {} seconds.".format(
                    str(best_f1), best_checkpoint, str(second_best_f1), second_best_checkpoint, args.eval_sleep))
                if not args.is_debug:
                    dict_to_log = {}
                    dict_to_log['eval_best_f1'] = best_f1
                    dict_to_log['eval_best_f1_checkpoint'] = best_checkpoint
                    dict_to_log['eval_second_best_f1'] = second_best_f1
                    dict_to_log['eval_second_best_f1_checkpoint'] = second_best_checkpoint
                    wandb.log(dict_to_log, step=global_step)
                return True

def evaluate(args, eval_dataloader, eval_dataset, model, criterion, prefix="", coref_threshold=0.5, cluster_threshold=0.5, thresh_delta=0.02):  #TODO: use threshold when resuming from checkpoint rather than searching it
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num steps = %d", try_measure_len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    losses = []
    losses_parts = {}
    batch_sizes = []
    all_cluster_logits_cpu = []
    all_coref_logits_cpu = []
    all_mention_logits_cpu = []
    all_cluster_logits_cuda = []
    all_input_ids = []
    all_coref_logits_cuda = []
    all_mention_logits_cuda = []
    all_gold_clusters = []
    all_gold_mentions = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        sum_text_len = [sum(tl) for tl in batch['text_len']]
        gold_clusters = batch['clusters']


        gold_mentions_list = []
        # if len(gold_clusters) > 0: #TODO:
        gold_mentions_list = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
        if args.add_junk:
            gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, args.device)
        else:
            gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=args.device) for gm in gold_mentions_list]
        
        gold_matrix = create_gold_matrix(args.device, sum_text_len, args.num_queries, gold_clusters, gold_mentions_list)

        input_ids, input_mask, sum_text_len, gold_mentions, num_mentions = tensor_and_remove_empty(batch, gold_mentions_list, args)
        if len(input_ids) == 0:
            continue
        max_mentions = torch.tensor(gold_mentions.shape[1]) if args.use_gold_mentions else sum_text_len.max()//2
        max_mentions = max_mentions.repeat([input_ids.shape[0], 1])
            
        all_gold_mentions += gold_mentions_list
        all_input_ids += input_ids    
        all_gold_clusters += gold_clusters

        with torch.no_grad():
            # orig_input_dim = input_ids.shape
            # input_ids = torch.reshape(input_ids, (1, -1))
            # input_mask = torch.reshape(input_mask, (1, -1))
            outputs = model(input_ids, max_mentions, input_mask, gold_mentions, num_mentions)
            cluster_logits, coref_logits, mention_logits = outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

            loss, loss_parts = criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})
            losses.append(loss.mean().detach().cpu())
            for key in loss_parts.keys():
                if key in losses_parts.keys() and len(losses_parts[key]) > 0:
                    losses_parts[key] += loss_parts[key]
                else:
                    losses_parts[key] = loss_parts[key]
            batch_sizes.append(loss.shape[0]) 

        if args.add_junk:
            all_mention_logits_cuda += [ml.detach().clone() for ml in mention_logits]
            all_mention_logits_cpu += [ml.detach().cpu() for ml in mention_logits]
        all_cluster_logits_cuda += [cl.detach().clone() for cl in cluster_logits]
        all_coref_logits_cuda += [cl.detach().clone() for cl in coref_logits]
        all_cluster_logits_cpu += [cl.detach().cpu() for cl in cluster_logits]
        all_coref_logits_cpu += [cl.detach().cpu() for cl in coref_logits]

    eval_loss = np.average(losses, weights=batch_sizes)
    losses_parts = {key:np.average(losses_parts[key]) for key in losses_parts.keys()}

    p, r, f1, best_coref_threshold, best_cluster_threshold, metrics = calc_best_avg_f1(all_cluster_logits_cpu, all_coref_logits_cpu, all_mention_logits_cpu, all_gold_clusters, all_gold_mentions, coref_threshold, cluster_threshold, thresh_delta)

    print_predictions(all_cluster_logits_cuda, all_coref_logits_cuda, all_mention_logits_cuda, all_gold_clusters, all_gold_mentions, all_input_ids, coref_threshold, cluster_threshold, args, eval_dataset.tokenizer)
    prec_gold_to_one_pred, prec_pred_to_one_gold, avg_gold_split_without_perfect, avg_gold_split_with_perfect, \
        avg_pred_split_without_perfect, avg_pred_split_with_perfect, prec_biggest_gold_in_pred_without_perfect, \
            prec_biggest_gold_in_pred_with_perfect, prec_biggest_pred_in_gold_without_perfect, prec_biggest_pred_in_gold_with_perfect = \
                error_analysis(all_cluster_logits_cuda, all_coref_logits_cuda, all_mention_logits_cuda, all_gold_clusters, all_gold_mentions, all_input_ids, coref_threshold, cluster_threshold)

    results = {'loss': eval_loss,
               'avg_f1': f1,
               'coref_threshold': best_coref_threshold, 
               'cluster_threshold': best_cluster_threshold,
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
               'prec_correct_mentions': metrics[0],
               'prec_gold': metrics[1],
               'prec_junk': metrics[2],
               'prec_correct_gold_clusters': metrics[3],
               'prec_correct_predict_clusters': metrics[4]} | losses_parts

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:

        def out(s):
            logger.info(str(s))
            writer.write(str(s) + '\n')

        out("***** Eval results {} *****".format(prefix))

        for key in sorted(results.keys()):
            out("eval %s = %s" % (key, str(results[key])))
        
    return results
