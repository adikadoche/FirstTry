import random
import os
from datetime import datetime
from time import time
from typing import List
# import git
import itertools
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from metrics import CorefEvaluator

from consts import NULL_ID_FOR_COREF

import logging
logger = logging.getLogger(__name__)



def save_checkpoint(args, global_step, threshold, model, optimizer, amp=None):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, 'threshold-{}_checkpoint-{}'.format(threshold, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(output_dir)
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.step-{}.pt'.format(global_step)))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
    if amp is not None:
        torch.save(amp.state_dict(), os.path.join(output_dir, 'amp.pt'))
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saved model checkpoint to %s", output_dir)


def load_from_checkpoint(model, checkpoint_path, device=None, optimizer=None, amp=None):
    global_step = checkpoint_path.rstrip('/').split('-')[-1]
    threshold = float(checkpoint_path.rstrip('/').split('-')[-2].split('_')[0])
    model_to_load = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    try:
        model_to_load.load_state_dict(torch.load(checkpoint_path + '/model.step-' + global_step + '.pt', map_location=device))
    except Exception as e:
        logger.error(e)
        model_to_load.load_state_dict(
            torch.load(checkpoint_path + '/model.step-' + global_step + '.pt', map_location=device), strict=False)
    if optimizer is not None:
        opt_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            # TODO make this more robust for different trees of states
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
    if amp is not None:
        amp.load_state_dict(torch.load(os.path.join(checkpoint_path, 'amp.pt')))
    return {'global_step':global_step, 'threshold':threshold}


def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters


def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t


# def write_meta_data(output_dir, args):
#     output_path = os.path.join(output_dir, "meta.json")
#     repo = git.Repo(search_parent_directories=True)
#     hexsha = repo.head.commit.hexsha
#     ts = time()
#     print(f"Writing {output_path}")
#     with open(output_path, mode='w') as f:
#         json.dump(
#             {
#                 'git_hexsha': hexsha,
#                 'args': {k: str(v) for k, v in args.__dict__.items()},
#                 'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#             },
#             f,
#             indent=4,
#             sort_keys=True)
#         print(file=f)



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def create_gold_matrix(device, doc_len, num_queries, gold_clusters, gold_mentions: List):
    if gold_mentions is None:
        gold_per_token = torch.zeros(num_queries, doc_len, device=device)
        for cluster_id, cluster in enumerate(gold_clusters):
            for start, end in cluster:
                gold_per_token[cluster_id, start: end + 1] = 1
    else:
        gold_per_token = torch.zeros(num_queries, len(gold_mentions), device=device)
        if num_queries < len(gold_clusters):
            logger.info("in utils, exceeds num_queries with length {}".format(len(gold_clusters)))
        for cluster_id, cluster in enumerate(gold_clusters):
            if cluster_id >= num_queries:
                continue
            for mention in cluster:
                mention_index = gold_mentions.index(tuple(mention))
                assert mention_index >= 0
                gold_per_token[cluster_id, mention_index] = 1

    return gold_per_token

def make_mentions_from_clustered_tokens(self, coref_logits):
    pass

def calc_predicted_clusters(cluster_logits, coref_logits, threshold, gold_mentions: List):
    # when we are using gold mentions, we get coref_logits at the size of the gold mentions ([1, clusters, gold_mentions]) (because we know they are mentions, what we are predicting is the clustering)
    coref_logits = coref_logits.squeeze(0) #[clusters, gold_mentions]

    if gold_mentions is None:
        cluster_bools = cluster_logits.numpy() >= threshold #TODO: should the cluster and coref share the same threshold?
        coref_bools = coref_logits.numpy() >= threshold

        true_coref_indices = np.asarray(np.where(coref_bools)).T
        cluster_id_to_tokens = {k: list(v) for k, v in itertools.groupby(sorted(true_coref_indices, key=lambda x: x[-1]), lambda x: x[-1])}

        bsz, num_of_clusters, _ = coref_bools.shape
        clusters = []
        for i in range(0, num_of_clusters):
            current_cluster = []

            first_token_index = None
            for token_index, token_logit in enumerate(coref_bools[0, i]):
                if token_logit:
                    if first_token_index is None:
                        first_token_index = token_index
                elif first_token_index is not None:
                    current_cluster.append((first_token_index, token_index-1))
                    first_token_index = None

            if first_token_index is not None:
                current_cluster.append((first_token_index, token_index))

            if len(current_cluster) > 0:
                clusters.append(current_cluster)
    else:
        cluster_bools = cluster_logits.squeeze().numpy() >= threshold #TODO: should the cluster and coref share the same threshold?
        cluster_bools = np.tile(cluster_bools.reshape([-1, 1]), (1, coref_logits.shape[1]))
        cluster_bools_mask = cluster_bools.astype(int)
        
        coref_logits_after_cluster_bool = np.multiply(cluster_bools_mask, coref_logits)
        max_coref_score, max_coref_cluster_ind = coref_logits_after_cluster_bool.max(0) #[gold_mention] choosing the index of the best cluster per gold mention
        coref_bools = max_coref_score >= threshold #[gold_mention] is the chosen cluster's score passes the threshold
        true_coref_indices = np.where(coref_bools)[0] #indices of the gold mention that their clusters pass threshold
        max_coref_cluster_ind = max_coref_cluster_ind[coref_bools] #index of the best clusters per gold mention, if it passes the threshold

        cluster_id_to_tokens = {k: list(v) for k, v in itertools.groupby(sorted(list(zip(true_coref_indices, max_coref_cluster_ind.numpy())), key=lambda x: x[-1]), lambda x: x[-1])}

        clusters = []

        for gold_mentions_inds in cluster_id_to_tokens.values():
            current_cluster = []
            for mention_id in gold_mentions_inds:
                try:
                    current_cluster.append(gold_mentions[mention_id[0]])
                except:
                    print('here')
            clusters.append(current_cluster)

    return clusters

def calc_best_avg_f1(all_cluster_logits, all_coref_logits, all_gold_clusters, all_gold_mentions):
    best = [-1, -1, -1]
    best_metrics = []
    best_threshold = None
    thres_start = 0.05
    thres_stop = 1
    thres_step = 0.05
    for threshold in tqdm(np.arange(thres_start, thres_stop, thres_step), desc='Searching for best threshold'):
        p, r, f1, metrics = evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_gold_clusters, threshold, all_gold_mentions)
        if f1 > best[-1]:
            best = p,r,f1
            best_metrics = metrics
            best_threshold = threshold

    return best + (best_threshold,) + (best_metrics,)

def evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_gold_clusters, threshold, all_gold_mentions):
    evaluator = CorefEvaluator()
    metrics = [0] * 5
    for i, (cluster_logits, coref_logits, gold_clusters, gold_mentions) in enumerate(
            zip(all_cluster_logits, all_coref_logits, all_gold_clusters, all_gold_mentions)):
        predicted_clusters = calc_predicted_clusters(cluster_logits, coref_logits, threshold, gold_mentions)
        prec_correct_mentions, prec_gold, prec_junk, prec_correct_gold_clusters, prec_correct_predict_clusters = \
            get_more_metrics(predicted_clusters, gold_clusters, gold_mentions)
        metrics[0] += prec_correct_mentions / len(all_cluster_logits)
        metrics[1] += prec_gold / len(all_cluster_logits)
        metrics[2] += prec_junk / len(all_cluster_logits)
        metrics[3] += prec_correct_gold_clusters / len(all_cluster_logits)
        metrics[4] += prec_correct_predict_clusters / len(all_cluster_logits)
        evaluator.update(predicted_clusters, gold_clusters)
    p, r, f1 = evaluator.get_prf()
    return p, r, f1, metrics

def get_more_metrics(predicted_clusters, gold_clusters, gold_mentions):
    prec_correct_mentions, prec_gold, prec_junk, prec_correct_gold_clusters, prec_correct_predict_clusters = 0,0,0,0,0
    real_gold_mentions = [m for c in gold_clusters for m in c]
    junk_gold_mentions = [m for m in gold_mentions if m not in real_gold_mentions]
    predicted_mentions = [m for c in predicted_clusters for m in c]

    if len(predicted_mentions)==0:
        if len(real_gold_mentions)>0:
            prec_correct_mentions = 0
            prec_gold = 0
        else:
            prec_correct_mentions = 1
            prec_gold = 1
        if len(junk_gold_mentions)>0:
            prec_junk = 0
        else:
            prec_junk = 1
    else:
        prec_correct_mentions = len([m for m in predicted_mentions if m in real_gold_mentions]) / len(predicted_mentions)
        if len(real_gold_mentions) == 0:
            prec_gold = 1
        else:
            prec_gold = len([m for m in real_gold_mentions if m in predicted_mentions]) / len(real_gold_mentions)
        if len(junk_gold_mentions) == 0:
            prec_junk = 1
        else:
            prec_junk = len([m for m in junk_gold_mentions if m in predicted_mentions]) / len(junk_gold_mentions)

    if len(predicted_clusters) == 0:
        if len(gold_clusters) > 0:
            prec_correct_gold_clusters = 0
            prec_correct_predict_clusters = 0
        else:
            prec_correct_gold_clusters = 1
            prec_correct_predict_clusters = 1
    else:
        prec_correct_predict_clusters = len([c for c in predicted_clusters if c in gold_clusters]) / len(predicted_clusters)
        if len(gold_clusters) > 0:
            prec_correct_gold_clusters = len([c for c in gold_clusters if c in predicted_clusters]) / len(gold_clusters)
        else:
            prec_correct_gold_clusters = 1

    return prec_correct_mentions, prec_gold, prec_junk, prec_correct_gold_clusters, prec_correct_predict_clusters

def try_measure_len(iter):
    try:
        return len(iter)
    except:
        return -1



def create_junk_gold_mentions(gold_mentions, text_len):
    num_mentions = len(gold_mentions)
    if num_mentions == 0:
        num_junk_mentions = random.randint(0, int(text_len/5))
    else:
        num_junk_mentions = random.randint(0, min(num_mentions*2, int(text_len/5)))

    junk_start_indices = np.random.permutation(range(text_len))[:num_junk_mentions]
    junk_end_indices = [min(start + random.randint(0, 5), text_len-1) for start in junk_start_indices]
    only_junk_mentions = [tuple((junk_start_indices[i], junk_end_indices[i])) for i in range(len(junk_start_indices))]
    only_junk_mentions = [f for f in only_junk_mentions if f not in gold_mentions]

    all_mentions = gold_mentions + only_junk_mentions
    random.shuffle(all_mentions)

    return all_mentions