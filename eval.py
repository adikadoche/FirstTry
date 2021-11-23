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
from utils import tensor_and_remove_empty, calc_predicted_clusters, create_gold_matrix, try_measure_len, load_from_checkpoint, create_junk_gold_mentions
from conll import evaluate_conll
from consts import TOKENS_PAD, SPEAKER_PAD
from metrics import CorefEvaluator
from matcher import build_matcher
import wandb
logger = logging.getLogger(__name__)



def report_eval(args, eval_dataloader, eval_dataset, global_step, model, criterion, threshold):
    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(args, eval_dataloader, eval_dataset, model, criterion, str(global_step), threshold)
        for key, value in results.items():
            wandb.log({'eval_{}'.format(key): value}, step=global_step)
        return results
    return None

def make_evaluation(model, criterion, eval_loader, eval_dataset, args):  
    # Evaluation 'no', 'specific', 'all', 'vanilla'
    if args.eval == 'specific':
        checkpoint = args.output_dir
        loaded_args = load_from_checkpoint(model, checkpoint, args)
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
                        loaded_args = load_from_checkpoint(model, checkpoint, args)
                        global_step = int(loaded_args['global_step'])
                        threshold = loaded_args['threshold']
                        results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, threshold)
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
                wandb.log({'eval_best_f1': best_f1})
                wandb.log({'eval_best_f1_checkpoint': best_checkpoint})
                wandb.log({'eval_second_best_f1': second_best_f1})
                wandb.log({'eval_second_best_f1_checkpoint': second_best_checkpoint})
                return True

def evaluate(args, eval_dataloader, eval_dataset, model, criterion, prefix="", threshold=0.5):  #TODO: use threshold when resuming from checkpoint rather than searching it
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num steps = %d", try_measure_len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    best_losses = ()
    best_all_cluster_logits_cuda = []
    best_all_coref_logits_cuda = []
    best_all_mention_logits_cuda = []
    all_input_ids = []
    all_gold_clusters = []
    all_gold_mentions = []

    input_ids_pads = torch.ones(1, args.max_segment_len, dtype=torch.int, device=args.device) * TOKENS_PAD
    mask_pads = torch.zeros(1, args.max_segment_len, dtype=torch.int, device=args.device)
    speaker_ids_pads = torch.ones(1, args.max_segment_len, args.max_num_speakers, dtype=torch.int, device=args.device) * SPEAKER_PAD

    best = [-1, -1, -1]
    best_metrics = []
    best_threshold = None
    thres_start = 0.05
    thres_stop = 1
    thres_step = 0.2
    # thress = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    thress = [0.05]

    hung_matcher = build_matcher(args)

    print('Searching for best threshold')
    # for threshold in np.arange(thres_start, thres_stop, thres_step):
    for threshold in thress:
        print(f'Checking threshold {threshold}')
        all_cluster_logits_cuda = []
        all_coref_logits_cuda = []
        all_mention_logits_cuda = []
        losses = []
        losses_parts = {}
        query_cluster_confusion_matrix = np.zeros([args.num_queries, args.num_queries], dtype=int)
        batch_sizes = []
        evaluator = CorefEvaluator()
        metrics = [0] * 5
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

            input_ids, input_mask, sum_text_len, gold_mentions, num_mentions, speaker_ids, genre = tensor_and_remove_empty(batch, gold_mentions_list, args, input_ids_pads, mask_pads, speaker_ids_pads)
            if len(input_ids) == 0:
                continue
            
            if threshold == thres_start:
                all_gold_mentions += gold_mentions_list
                all_input_ids += input_ids    
                all_gold_clusters += gold_clusters

            with torch.no_grad():
                mention_logits = []
                outputs = model(input_ids, input_mask, gold_mentions, num_mentions, speaker_ids, genre)
                cluster_logits, coref_logits = outputs['cluster_logits'], outputs['coref_logits']
                
                if args.add_junk:
                    predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
                                                                threshold, gold_mentions_list, args.is_max)
                else:
                    predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                                threshold, gold_mentions_list, args.is_max)
                evaluator.update(predicted_clusters, gold_clusters)
                targets = {'clusters':gold_matrix, 'mentions':gold_mentions_vector}
                matched_predicted_cluster_id, matched_gold_cluster_id = hung_matcher(outputs, targets)
                for i, j in zip(matched_predicted_cluster_id[0], matched_gold_cluster_id[0]):
                    query_cluster_confusion_matrix[i][j] += 1
                loss, loss_parts = criterion(outputs, targets)
                losses.append(loss.mean().detach().cpu())
                for key in loss_parts.keys():
                    if key in losses_parts.keys() and len(losses_parts[key]) > 0:
                        losses_parts[key] += loss_parts[key]
                    else:
                        losses_parts[key] = loss_parts[key]
                batch_sizes.append(loss.shape[0]) 
            if args.add_junk:
                all_mention_logits_cuda += [ml.detach().clone() for ml in mention_logits]
            all_cluster_logits_cuda += [cl.detach().clone() for cl in cluster_logits]
            all_coref_logits_cuda += [cl.detach().clone() for cl in coref_logits]
        p, r, f1 = evaluator.get_prf()
        if f1 > best[-1]:
            best = p,r,f1
            best_metrics = metrics
            best_threshold = threshold
            best_all_mention_logits_cuda = all_mention_logits_cuda
            best_all_cluster_logits_cuda = all_cluster_logits_cuda
            best_all_coref_logits_cuda = all_coref_logits_cuda
            best_losses = (losses, losses_parts, batch_sizes)

    losses = best_losses[0]
    losses_parts = best_losses[1]
    batch_sizes = best_losses[2]
    eval_loss = np.average(losses, weights=batch_sizes)
    losses_parts = {key:np.average(losses_parts[key]) for key in losses_parts.keys()}
    threshold = best_threshold
    metrics = best_metrics
    p, r, f1 = best


    print_predictions(best_all_cluster_logits_cuda, best_all_coref_logits_cuda, best_all_mention_logits_cuda, \
        all_gold_clusters, all_gold_mentions, all_input_ids, threshold, args, eval_dataset.tokenizer)
    prec_gold_to_one_pred, prec_pred_to_one_gold, avg_gold_split_without_perfect, avg_gold_split_with_perfect, \
        avg_pred_split_without_perfect, avg_pred_split_with_perfect, prec_biggest_gold_in_pred_without_perfect, \
            prec_biggest_gold_in_pred_with_perfect, prec_biggest_pred_in_gold_without_perfect, prec_biggest_pred_in_gold_with_perfect = \
                error_analysis(best_all_cluster_logits_cuda, best_all_coref_logits_cuda, best_all_mention_logits_cuda, all_gold_clusters, \
                    all_gold_mentions, all_input_ids, threshold, args.is_max)

    non_zero_rows = np.where(np.sum(query_cluster_confusion_matrix, 1) > 0)[0]
    non_zero_cols = np.where(np.sum(query_cluster_confusion_matrix, 0) > 0)[0]
    print('rows - predict, cols - gt')
    line_to_print = '   '
    for i in non_zero_cols:
        line_to_print += str(i) + ' ' + ('' if i>=10 else ' ')
    print(line_to_print)
    for i in non_zero_rows:
        line_to_print = str(i) + ' ' + ('' if i>=10 else ' ')
        for j in non_zero_cols:
            line_to_print += str(query_cluster_confusion_matrix[i][j]) + ' ' + ('' if query_cluster_confusion_matrix[i][j]>=10 else ' ')
        print(line_to_print)
    


    results = {'loss': eval_loss,
               'avg_f1': f1,
               'threshold': threshold,
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
