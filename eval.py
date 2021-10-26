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
from coref_analysis import print_predictions, print_per_batch
from data import get_dataset
from utils import calc_best_avg_f1, create_gold_matrix, try_measure_len, load_from_checkpoint, create_junk_gold_mentions
from conll import evaluate_conll
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
        loaded_args = load_from_checkpoint(model, checkpoint)
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
                        loaded_args = load_from_checkpoint(model, checkpoint)
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
                # time.sleep(args.eval_sleep)
                return True

def tensor_and_remove_empty(batch, gold_mentions, args):
    input_ids, input_mask, sum_text_len, gold_clusters, new_gold_mentions = [], [], [], [], []
    for i in range(len(gold_mentions)):
        if len(gold_mentions[i]) > 0:
            input_ids.append(torch.tensor(batch['input_ids'][i]).to(args.device))
            input_mask.append(torch.tensor(batch['input_mask'][i]).to(args.device))
            sum_text_len.append(sum(batch['text_len'][i]))
            gold_clusters.append(batch['clusters'][i])
            new_gold_mentions.append(gold_mentions[i])
    return input_ids, input_mask, sum_text_len, gold_clusters, new_gold_mentions


def evaluate(args, eval_dataloader, eval_dataset, model, criterion, prefix="", threshold=0.5):  #TODO: use threshold when resuming from checkpoint rather than searching it
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

    count_clusters = 0
    count_mentions = 0
    
    count_pronouns_mentions = 0
    count_clusters_with_pronoun_mention = 0
    
    count_missed_mentions = 0
    count_missed_pronouns = 0
    count_excess_mentions = 0
    count_excess_pronous = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        sum_text_len = [sum(tl) for tl in batch['text_len']]
        gold_clusters = batch['clusters']


        gold_mentions = []
        # if len(gold_clusters) > 0: #TODO:
        gold_mentions = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
        if args.add_junk:
            gold_mentions, gold_mentions_vector = create_junk_gold_mentions(gold_mentions, sum_text_len, args.device)
        else:
            gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=args.device) for gm in gold_mentions]
        
        input_ids, input_mask, sum_text_len, gold_clusters, gold_mentions = tensor_and_remove_empty(batch, gold_mentions, args)
        if len(input_ids) == 0:
            continue
            
        gold_matrix = create_gold_matrix(args.device, sum_text_len, args.num_queries, gold_clusters, gold_mentions)

        all_gold_mentions += gold_mentions
        all_input_ids += input_ids    
        all_gold_clusters += gold_clusters

        with torch.no_grad():
            # orig_input_dim = input_ids.shape
            # input_ids = torch.reshape(input_ids, (1, -1))
            # input_mask = torch.reshape(input_mask, (1, -1))
            outputs = model(input_ids, sum_text_len, input_mask, gold_mentions)
            cluster_logits, coref_logits, mention_logits = outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

            loss, loss_parts = criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})
            losses.append(loss.mean().detach().cpu())
            for key in loss_parts.keys():
                if key in losses_parts.keys() and len(losses_parts[key]) > 0:
                    losses_parts[key] += loss_parts[key]
                else:
                    losses_parts[key] = loss_parts[key]
            batch_sizes.append(loss.shape[0]) 

        all_mention_logits_cuda += [ml.detach().clone() for ml in mention_logits]
        all_cluster_logits_cuda += [cl.detach().clone() for cl in cluster_logits]
        all_coref_logits_cuda += [cl.detach().clone() for cl in coref_logits]
        all_mention_logits_cpu += [ml.detach().cpu() for ml in mention_logits]
        all_cluster_logits_cpu += [cl.detach().cpu() for cl in cluster_logits]
        all_coref_logits_cpu += [cl.detach().cpu() for cl in coref_logits]

    eval_loss = np.average(losses, weights=batch_sizes)
    losses_parts = {key:np.average(losses_parts[key]) for key in losses_parts.keys()}

    p, r, f1, threshold, metrics = calc_best_avg_f1(all_cluster_logits_cpu, all_coref_logits_cpu, all_mention_logits_cpu, all_gold_clusters, all_gold_mentions)
    results = {'loss': eval_loss,
               'avg_f1': f1,
               'threshold': threshold,
               'precision': p,
               'recall': r,  
               'prec_correct_mentions': metrics[0],
               'prec_gold': metrics[1],
               'prec_junk': metrics[2],
               'prec_correct_gold_clusters': metrics[3],
               'prec_correct_predict_clusters': metrics[4]} | losses_parts


    print_predictions(all_cluster_logits_cuda, all_coref_logits_cuda, all_mention_logits_cuda, all_gold_clusters, all_gold_mentions, all_input_ids, threshold, args, eval_dataset.tokenizer)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:

        def out(s):
            logger.info(str(s))
            writer.write(str(s) + '\n')

        out("***** Eval results {} *****".format(prefix))

        for key in sorted(results.keys()):
            out("eval %s = %s" % (key, str(results[key])))

    return results
