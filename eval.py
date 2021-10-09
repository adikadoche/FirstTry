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
from utils import calc_best_avg_f1, create_gold_matrix, try_measure_len, load_from_checkpoint
from conll import evaluate_conll
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)



def report_eval(args, eval_dataloader, eval_dataset, global_step, model, criterion, tb_writer, threshold):
    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(args, eval_dataloader, eval_dataset, model, criterion, str(global_step), threshold)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
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
        tb_writer = SummaryWriter(log_dir=args.output_dir, flush_secs=15)

        evaluated = set()
        while True:
            if args.resume_from and not args.do_train:
                checkpoints = list(
                    os.path.dirname(c) for c in glob.glob(original_output_dir + '/model*.pt', recursive=True))
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
                        global_step = loaded_args['global_step']
                        threshold = loaded_args['threshold']
                        report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, tb_writer, threshold)

                    evaluated.update(checkpoints)
                except Exception as e:
                    logger.error(
                        "Got an exception. Will sleep for {} and try again. {}".format(args.eval_sleep, repr(e)))
                    time.sleep(args.eval_sleep)
            else:
                logger.info("No new checkpoints. Sleep for {} seconds.".format(args.eval_sleep))
                time.sleep(args.eval_sleep)



def evaluate(args, eval_dataloader, eval_dataset, model, criterion, prefix="", threshold=0.5):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num steps = %d", try_measure_len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    losses = []
    batch_sizes = []
    all_cluster_logits_cpu = []
    all_coref_logits_cpu = []
    all_cluster_logits_cuda = []
    all_coref_logits_cuda = []
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

        batch = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch)
        input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, sentence_map, gold_clusters = batch
        
        if len(gold_clusters) == 0 or len(gold_clusters) > args.num_queries:
            logger.info("eval exceeds num_queries with length {}".format(len(gold_clusters)))

        all_gold_clusters.append(gold_clusters)

        gold_mentions = []
        if args.use_gold_mentions and len(gold_clusters) > 0:
            gold_mentions = list(set([tuple(m) for c in gold_clusters for m in c]))
        all_gold_mentions.append(gold_mentions)

        with torch.no_grad():
            orig_input_dim = input_ids.shape
            input_ids = torch.reshape(input_ids, (1, -1))
            input_mask = torch.reshape(input_mask, (1, -1))
            outputs = model(input_ids, orig_input_dim, input_mask, gold_mentions)
            cluster_logits, coref_logits = outputs['cluster_logits'], outputs['coref_logits']

            # count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, \
            #     count_missed_mentions, count_missed_pronouns, count_excess_pronous, count_excess_mentions = print_per_batch(0, True,
            #     cluster_logits, coref_logits, threshold, gold_clusters, gold_mentions, eval_dataset, input_ids,
            #     count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, count_missed_mentions,
            #     count_missed_pronouns, count_excess_pronous, count_excess_mentions)


            gold_matrix = create_gold_matrix(args.device, text_len.sum(), args.num_queries, gold_clusters, gold_mentions)
            loss = criterion(outputs, gold_matrix)
            losses.append(loss.item())
            batch_sizes.append(1) # TODO support batches

        all_cluster_logits_cuda.append(cluster_logits.detach().clone())
        all_coref_logits_cuda.append(coref_logits.detach().clone())
        all_cluster_logits_cpu.append(cluster_logits.detach().cpu())
        all_coref_logits_cpu.append(coref_logits.detach().cpu())

    eval_loss = np.average(losses, weights=batch_sizes)

    p, r, f1, threshold = calc_best_avg_f1(all_cluster_logits_cpu, all_coref_logits_cpu, all_gold_clusters, all_gold_mentions)
    results = {'loss': eval_loss,
               'avg_f1': f1,
               'threshold': threshold,
               'precision': p,
               'recall': r}


    print_predictions(all_cluster_logits_cuda, all_coref_logits_cuda, eval_dataloader, eval_dataset, model, threshold, args) #TODO: save all the results in the loop above and send it to the function so it wont have to call model itself


    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:

        def out(s):
            logger.info(str(s))
            writer.write(str(s) + '\n')

        out("***** Eval results {} *****".format(prefix))

        for key in sorted(results.keys()):
            out("eval %s = %s" % (key, str(results[key])))

    return results
