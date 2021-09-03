import json
import os
import logging
from collections import Counter
import random
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
from tqdm import tqdm
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from utils import calc_best_avg_f1, create_gold_matrix, try_measure_len
from conll import evaluate_conll

logger = logging.getLogger(__name__)


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem

def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)




def evaluate(args, eval_dataloader, model, criterion, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num steps = %d", try_measure_len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    losses = []
    batch_sizes = []
    all_cluster_logits = []
    all_coref_logits = []
    all_gold_clusters = []
    all_gold_mentions = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch)
        input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, sentence_map, gold_clusters = batch
        all_gold_clusters.append(gold_clusters)

        gold_mentions = None
        if args.use_gold_mentions:
            gold_mentions = list(set([tuple(m) for c in gold_clusters for m in c]))
        all_gold_mentions.append(gold_mentions)

        with torch.no_grad():
            outputs = model(input_ids, input_mask, gold_mentions)
            cluster_logits, coref_logits = outputs['cluster_logits'], outputs['coref_logits']
            gold_matrix = create_gold_matrix(args.device, text_len.sum(), args.num_queries, gold_clusters, gold_mentions)
            loss = criterion(outputs, gold_matrix)
            losses.append(loss.item())
            batch_sizes.append(1) # TODO support batches

        all_cluster_logits.append(cluster_logits.detach().cpu())
        all_coref_logits.append(coref_logits.detach().cpu())

    eval_loss = np.average(losses, weights=batch_sizes)

    p, r, f1, threshold = calc_best_avg_f1(all_cluster_logits, all_coref_logits, all_gold_clusters, all_gold_mentions)
    results = {'loss': eval_loss,
               'avg_f1': f1,
               'threshold': threshold,
               'precision': p,
               'recall': r}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:

        def out(s):
            logger.info(str(s))
            writer.write(str(s) + '\n')

        out("***** Eval results {} *****".format(prefix))

        for key in sorted(results.keys()):
            out("eval %s = %s" % (key, str(results[key])))

    return results

