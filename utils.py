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

from consts import NULL_ID_FOR_COREF, TOKENS_PAD, MASK_PAD

import logging
logger = logging.getLogger(__name__)


# def save_checkpoint(args, global_step, numbers, model, optimizer, lr_scheduler, output_dir, amp=None):
#     # Save model checkpoint
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#     torch.save({
#         'model': model_to_save.state_dict(),
#         'coref_threshold': numbers['coref_threshold'],
#         'cluster_threshold': numbers['cluster_threshold'],
#         'thresh_delta': numbers['thresh_delta'],
#         'best_f1_global_step': numbers['best_f1_global_step'],
#         'last_saved_global_step': numbers['last_saved_global_step'],
#         'best_f1': numbers['best_f1'],
#         'optimizer': optimizer.state_dict(),
#         'lr_scheduler': lr_scheduler.state_dict(),
#         'args': args
#         }, os.path.join(output_dir, 'model.step-{}.pt'.format(global_step)))
#     logger.info("Saved model checkpoint to %s", output_dir)

def save_checkpoint(args, global_step, numbers, model, optimizer, lr_scheduler, output_dir, amp=None):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    torch.save({
        'model': model_to_save.state_dict(),
        'numbers': numbers,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args
        }, os.path.join(output_dir, 'model.step-{}.pt'.format(global_step)))
    logger.info("Saved model checkpoint to %s", output_dir)


def load_from_checkpoint(model, resume_from, device=None, optimizer=None, lr_scheduler=None, amp=None):
    global_step = resume_from.rstrip('/').split('-')[-1]
    checkpoint = torch.load(resume_from + '/model.step-' + global_step + '.pt', map_location=device)
    model_to_load = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_load.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    numbers = checkpoint['numbers']
            
    return {'global_step':global_step, 'numbers':numbers}

# def load_from_checkpoint(model, resume_from, device=None, optimizer=None, lr_scheduler=None, amp=None):
#     global_step = resume_from.rstrip('/').split('-')[-1]
#     checkpoint = torch.load(resume_from + '/model.step-' + global_step + '.pt', map_location=device)
#     model_to_load = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#     model_to_load.load_state_dict(checkpoint['model'])
#     for _ in range(int(global_step)):
#         lr_scheduler.step()
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     coref_threshold = checkpoint['coref_threshold']
#     cluster_threshold = checkpoint['cluster_threshold']
            
#     return {'global_step':global_step, 'coref_threshold':coref_threshold, 'cluster_threshold':cluster_threshold}


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

def create_gold_matrix(device, doc_len, num_queries, gold_clusters, gold_mentions: List, max_mentions):
    if gold_mentions is None:
        gold_per_token = torch.zeros(num_queries, doc_len, device=device)
        for cluster_id, cluster in enumerate(gold_clusters):
            for start, end in cluster:
                gold_per_token[cluster_id, start: end + 1] = 1
    else:
        gold_per_token = torch.zeros(len(gold_mentions), num_queries, max_mentions, device=device)
        for i in range(len(gold_clusters)):
            if num_queries < len(gold_clusters[i]):
                logger.info("in utils, exceeds num_queries with length {}".format(len(gold_clusters[i])))
            for cluster_id, cluster in enumerate(gold_clusters[i]):
                if cluster_id >= num_queries:
                    continue
                for mention in cluster:
                    mention_index = gold_mentions[i].index(tuple(mention))
                    assert mention_index >= 0
                    gold_per_token[i, cluster_id, mention_index] = 1
            # if gold_per_token.shape[1] == 0:
            #     logger.info("size of gold_cluster {}, size of gold matrix {}".format(len(gold_clusters[i]), gold_per_token.shape))

    return gold_per_token

def create_target_and_predict_matrix(gold_mentions_list, mentions_list, gold_matrix):
    target_matrix_list = []
    is_gold_mention = []
    for b in range(len(gold_matrix)):
        junk_mentions_indices = torch.tensor([i for i, m in enumerate(mentions_list[b]) if m not in gold_mentions_list[b]], dtype=torch.long, device=gold_matrix[0].device)
        common_mentions = [m for m in mentions_list[b] if m in gold_mentions_list[b]]

        common_gold_ind = torch.zeros(len(common_mentions), dtype=torch.long, device=gold_matrix[0].device)
        common_predict_ind = torch.zeros(len(common_mentions)+len(junk_mentions_indices), device=gold_matrix[0].device)

        ind = 0
        for i in range(len(mentions_list[b])):
            if mentions_list[b][i] in common_mentions:
                for j in range(len(gold_mentions_list[b])):
                    if gold_mentions_list[b][j] == mentions_list[b][i]:
                        common_gold_ind[ind] = j
                        common_predict_ind[i] = 1
                        ind += 1

        target_matrix = torch.zeros(len(common_mentions)+len(junk_mentions_indices), gold_matrix[b].shape[0], device=gold_matrix[b].device)
        target_matrix[common_predict_ind == 1] = torch.index_select(gold_matrix[b].transpose(0,1), 0, common_gold_ind)         
        target_matrix_list.append(target_matrix.transpose(0,1))
        is_gold_mention.append(common_predict_ind)
    return target_matrix_list, is_gold_mention

def make_mentions_from_clustered_tokens(self, coref_logits):
    pass

def calc_predicted_clusters(cluster_logits, coref_logits, mention_logits, coref_threshold, cluster_threshold, gold_mentions: List, slots=False):
    # when we are using gold mentions, we get coref_logits at the size of the gold mentions ([bs, clusters, gold_mentions]) (because we know they are mentions, what we are predicting is the clustering)
    
    bs = cluster_logits.shape[0]

    if gold_mentions is None:
        cluster_bools = cluster_logits.numpy() >= cluster_threshold #TODO: should the cluster and coref share the same threshold?
        coref_bools = coref_logits.numpy() >= coref_threshold

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
        cluster_bools = cluster_logits.squeeze(-1).numpy() >= cluster_threshold #TODO: should the cluster and coref share the same threshold?
        clusters = []
        for i in range(bs):
            cur_cluster_bool = cluster_bools[i]
            cur_coref_logits = coref_logits[i]
            cur_cluster_bool = np.tile(cur_cluster_bool.reshape([1, -1, 1]), (1, 1, cur_coref_logits.shape[-1]))
            cluster_mention_mask = cur_cluster_bool

            if slots:
                max_bools = torch.max(cur_coref_logits,0)[1].reshape([-1,1]).repeat([1, cur_coref_logits.shape[0]]) == \
                    torch.arange(cur_coref_logits.shape[0], device=cur_coref_logits.device).reshape([1, -1]).repeat(cur_coref_logits.shape[1], 1)
                max_bools = max_bools.transpose(0, 1).numpy()
                coref_bools = cluster_mention_mask & max_bools
                coref_logits_after_cluster_bool = np.multiply(coref_bools, cur_coref_logits)
                max_coref_score, max_coref_cluster_ind = coref_logits_after_cluster_bool[0].max(-2) #[gold_mention] choosing the index of the best cluster per gold mention
                coref_bools = max_coref_score > 0
            else:
                if len(mention_logits) > 0:
                    cur_mention_bools = mention_logits[i].squeeze(-1).numpy() >= cluster_threshold
                    cur_mention_bools = np.tile(cur_mention_bools.reshape([1, 1, -1]), (1, cur_cluster_bool.shape[1], 1))
                    cluster_mention_mask = cur_mention_bools & cur_cluster_bool
                cluster_mention_mask = cluster_mention_mask.astype(int)

                coref_logits_after_cluster_bool = np.multiply(cluster_mention_mask, cur_coref_logits.unsqueeze(0))
                max_coref_score, max_coref_cluster_ind = coref_logits_after_cluster_bool[0].max(-2) #[gold_mention] choosing the index of the best cluster per gold mention
                coref_bools = max_coref_score >= coref_threshold #[gold_mention] is the chosen cluster's score passes the threshold

            true_coref_indices = np.where(coref_bools)[0] #indices of the gold mention that their clusters pass threshold
            max_coref_cluster_ind_filtered = max_coref_cluster_ind[coref_bools] #index of the best clusters per gold mention, if it passes the threshold

            cluster_id_to_tokens = {k: list(v) for k, v in itertools.groupby(sorted(list(zip(true_coref_indices, max_coref_cluster_ind_filtered.numpy())), key=lambda x: x[-1]), lambda x: x[-1])}

            b_clusters = []

            for gold_mentions_inds in cluster_id_to_tokens.values():
                current_cluster = []
                for mention_id in gold_mentions_inds:
                    try:
                        current_cluster.append(gold_mentions[i][mention_id[0]])
                    except:
                        print('here')
                if len(current_cluster) > 1:
                    b_clusters.append(current_cluster)
            if len(b_clusters) > 1:
                clusters.append(b_clusters)

    if clusters == []:
        return [[]]
    return clusters

def calc_best_avg_f1(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, all_gold_mentions, coref_threshold, cluster_threshold, thresh_delta, slots):
    best = [-1, -1, -1, -1, -1, -1]
    best_metrics = []
    best_coref_threshold = None
    best_cluster_threshold = None
    if thresh_delta == 0.2:
        thresh_coref_start = 0.05
        thresh_coref_end = 1
        thresh_cluster_start = 0.05
        thresh_cluster_end = 1
    else:
        thresh_coref_start = max(0.01, coref_threshold - 2*thresh_delta)
        thresh_coref_end = min(1, coref_threshold + 2.5*thresh_delta)
        thresh_cluster_start = max(0.01, cluster_threshold - 2*thresh_delta)
        thresh_cluster_end = min(1, cluster_threshold + 2.5*thresh_delta)
    for coref_threshold in tqdm(np.arange(thresh_coref_start, thresh_coref_end, thresh_delta), desc='Searching for best threshold'):
        for cluster_threshold in np.arange(thresh_cluster_start, thresh_cluster_end, thresh_delta):
            p, r, f1, pm, rm, f1m, metrics = evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, coref_threshold, cluster_threshold, all_gold_mentions, slots)
            if f1 > best[-1]:
                best = pm, rm, f1m, p,r,f1
                best_metrics = metrics
                best_coref_threshold = coref_threshold
                best_cluster_threshold = cluster_threshold

    return best + (best_coref_threshold, best_cluster_threshold,) + (best_metrics,)

def evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, coref_threshold, cluster_threshold, all_gold_mentions, slots):
    cluster_evaluator = CorefEvaluator()
    mention_evaluator = CorefEvaluator()
    metrics = [0] * 5
    for i, (cluster_logits, coref_logits, gold_clusters, gold_mentions) in enumerate(
            zip(all_cluster_logits, all_coref_logits, all_gold_clusters, all_gold_mentions)):
        if len(all_mention_logits) > 0:
            mention_logits = all_mention_logits[i]
            predicted_clusters = calc_predicted_clusters(cluster_logits.unsqueeze(0), coref_logits.unsqueeze(0), mention_logits.unsqueeze(0), coref_threshold, cluster_threshold, [gold_mentions], slots)
        else:
            predicted_clusters = calc_predicted_clusters(cluster_logits.unsqueeze(0), coref_logits.unsqueeze(0), [], coref_threshold, cluster_threshold, [gold_mentions], slots)
        # prec_correct_mentions, prec_gold, prec_junk, prec_correct_gold_clusters, prec_correct_predict_clusters = \
        #     get_more_metrics(predicted_clusters, gold_clusters, gold_mentions)  #TODO: predicted_clusters[0]?
        # metrics[0] += prec_correct_mentions / len(all_cluster_logits)
        # metrics[1] += prec_gold / len(all_cluster_logits)
        # metrics[2] += prec_junk / len(all_cluster_logits)
        # metrics[3] += prec_correct_gold_clusters / len(all_cluster_logits)
        # metrics[4] += prec_correct_predict_clusters / len(all_cluster_logits)
        cluster_evaluator.update(predicted_clusters, [gold_clusters])
        gold_mentions_e = [[]] if [gold_clusters] == [[]] or [gold_clusters] == [()] else \
            [[[m for c in [gold_clusters] for d in c for m in d]]]
        predicted_mentions_e = [[]] if predicted_clusters == [[]] or predicted_clusters == [()] else [
            [[m for c in predicted_clusters for d in c for m in d]]]
        mention_evaluator.update(predicted_mentions_e, gold_mentions_e)
    p, r, f1 = cluster_evaluator.get_prf()
    pm, rm, f1m = mention_evaluator.get_prf()
    return p, r, f1, pm, rm, f1m, metrics

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



def create_junk_gold_mentions(gold_mentions, text_len, device):
    all_mentions = []
    real_mentions_bools = []
    for i in range(len(gold_mentions)):
        num_mentions = len(gold_mentions[i])
        if num_mentions == 0:
            num_junk_mentions = random.randint(0, int(text_len[i]/5))
        else:
            num_junk_mentions = random.randint(0, min(num_mentions*2, int(text_len[i]/5)))

        junk_start_indices = np.random.permutation(range(text_len[i]))[:num_junk_mentions]
        junk_end_indices = [min(start + random.randint(0, 5), text_len[i]-1) for start in junk_start_indices]
        only_junk_mentions = [tuple((junk_start_indices[i], junk_end_indices[i])) for i in range(len(junk_start_indices))]
        only_junk_mentions = [f for f in only_junk_mentions if f not in gold_mentions[i]]

        unite_mentions = gold_mentions[i] + only_junk_mentions
        indices = list(range(len(unite_mentions)))
        random.shuffle(indices)
        all_mentions.append([unite_mentions[ind] for ind in indices])
        real_mentions_bools.append(torch.tensor([int(ind < len(gold_mentions[i])) for ind in indices], dtype=torch.float, device=device))

    return all_mentions, real_mentions_bools


def pad_input_ids_and_mask_to_max_tokens(cur_input_ids, cur_mask, input_ids, mask, i):
    cur_input_ids = cur_input_ids.reshape(cur_input_ids.shape[0], -1)
    cur_mask = cur_mask.reshape(cur_mask.shape[0], -1)
    cur_input_ids = cur_input_ids[cur_mask==1]
    cur_mask = torch.ones_like(cur_input_ids)

    input_ids[i][:len(cur_input_ids)] = cur_input_ids
    mask[i][:len(cur_mask)] = cur_mask

    return input_ids, mask

def pad_mentions(gold_mentions, max_mentions, device):
    padded_gold_mentions = torch.tensor(np.asarray(gold_mentions + (max_mentions-len(gold_mentions)) * [(0, 0)])).unsqueeze(0)
    mask_gold_mentions = torch.cat([torch.ones(1, len(gold_mentions), dtype=torch.long, device=device), torch.zeros(1, max_mentions-len(gold_mentions), dtype=torch.long, device=device)], 1)
    return padded_gold_mentions, mask_gold_mentions

def tensor_and_remove_empty(batch, gold_mentions, args):
    bs = len(batch['text_len'])
    max_len = max([sum(batch['text_len'][i]) for i in range(len(batch['text_len']))])
    input_ids = torch.ones(bs, max_len, dtype=torch.int, device=args.device) * TOKENS_PAD
    input_mask = torch.ones(bs, max_len, dtype=torch.int, device=args.device) * MASK_PAD

    sum_text_len, num_mentions, new_gold_mentions, new_gold_mentions_mask = [], [], [], []
    max_mentions = max([len(gm) for gm in gold_mentions])
    for i in range(bs):
        input_ids, input_mask = pad_input_ids_and_mask_to_max_tokens(\
            torch.tensor(batch['input_ids'][i], device=args.device), torch.tensor(batch['input_mask'][i], device=args.device), input_ids, input_mask, i)
        sum_text_len.append(torch.tensor([sum(batch['text_len'][i])]).to(args.device))
        g_mention, g_mention_mask = pad_mentions(gold_mentions[i], max_mentions, args.device)
        new_gold_mentions.append(g_mention)
        new_gold_mentions_mask.append(g_mention_mask)
        num_mentions.append(torch.tensor([len(gold_mentions[i])]).to(args.device))
    return input_ids, input_mask, \
            torch.cat(sum_text_len).reshape(bs), \
                torch.cat(new_gold_mentions).reshape(bs, max_mentions, 2),\
                torch.cat(new_gold_mentions_mask).reshape(bs, max_mentions),\
                    torch.cat(num_mentions).reshape(bs)
