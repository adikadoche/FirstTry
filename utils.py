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



def save_checkpoint(args, global_step, threshold, model, optimizer, output_dir, amp=None):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(output_dir)
    torch.save({
        'model': model_to_save.state_dict(),
        'threshold': threshold,
        'optimizer': optimizer.state_dict(),
        'args': args
        }, os.path.join(output_dir, 'model.step-{}.pt'.format(global_step)))
    logger.info("Saved model checkpoint to %s", output_dir)


def load_from_checkpoint(model, checkpoint_path, args=None, optimizer=None, amp=None):
    global_step = checkpoint_path.rstrip('/').split('-')[-1]
    checkpoint = torch.load(checkpoint_path + '/model.step-' + global_step + '.pt', map_location=args.device)
    # if args is not None:
    #     args = checkpoint['args']
    model_to_load = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_load.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # TODO make this more robust for different trees of states
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
    threshold = checkpoint['threshold']
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
        gold_per_token_batch = []
        for i in range(len(gold_clusters)):
            gold_per_token = torch.zeros(num_queries, len(gold_mentions[i]), device=device)
            if num_queries < len(gold_clusters[i]):
                logger.info("in utils, exceeds num_queries with length {}".format(len(gold_clusters[i])))
            for cluster_id, cluster in enumerate(gold_clusters[i]):
                if cluster_id >= num_queries:
                    continue
                for mention in cluster:
                    mention_index = gold_mentions[i].index(tuple(mention))
                    assert mention_index >= 0
                    gold_per_token[cluster_id, mention_index] = 1
            # if gold_per_token.shape[1] == 0:
            #     logger.info("size of gold_cluster {}, size of gold matrix {}".format(len(gold_clusters[i]), gold_per_token.shape))
            gold_per_token_batch.append(gold_per_token)

    return gold_per_token_batch

def make_mentions_from_clustered_tokens(self, coref_logits):
    pass

def calc_predicted_clusters(cluster_logits, coref_logits, mention_logits, threshold, gold_mentions: List, is_max):
    # when we are using gold mentions, we get coref_logits at the size of the gold mentions ([bs, clusters, gold_mentions]) (because we know they are mentions, what we are predicting is the clustering)
    
    bs = coref_logits.shape[0]

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
        if is_max:
            cluster_bools = cluster_logits.squeeze(-1).numpy() >= 0.5
        else: 
            cluster_bools = cluster_logits.squeeze(-1).numpy() >= threshold #TODO: should the cluster and coref share the same threshold?
        clusters = []
        for i in range(bs):
            cur_cluster_bool = cluster_bools[i]
            cur_coref_logits = coref_logits[i]
            cur_cluster_bool = np.tile(cur_cluster_bool.reshape([1, -1, 1]), (1, 1, cur_coref_logits.shape[-1]))
            cluster_mention_mask = cur_cluster_bool
            if len(mention_logits) > 0:
                cur_mention_bools = mention_logits[i].squeeze(-1).numpy() >= threshold
                cur_mention_bools = np.tile(cur_mention_bools.reshape([1, 1, -1]), (1, cur_cluster_bool.shape[1], 1))
                cluster_mention_mask = cur_mention_bools & cluster_mention_mask             
            cluster_mention_mask = cluster_mention_mask.astype(int)
            
            coref_logits_after_cluster_bool = np.multiply(cluster_mention_mask, cur_coref_logits.unsqueeze(0))
            max_coref_score, max_coref_cluster_ind = coref_logits_after_cluster_bool[0].max(-2) #[gold_mention] choosing the index of the best cluster per gold mention
            if is_max:
                coref_bools = max_coref_score > 0
            else:
                coref_bools = max_coref_score >= threshold #[gold_mention] is the chosen cluster's score passes the threshold

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
                b_clusters.append(current_cluster)
            clusters.append(b_clusters)

    return clusters

def calc_best_avg_f1(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, all_gold_mentions, all_predicted_clusters, threshold):
    best = [-1, -1, -1]
    best_metrics = []
    best_threshold = None
    thres_start = 0.05
    thres_stop = 1
    thres_step = 0.05
    # for threshold in tqdm(np.arange(thres_start, thres_stop, thres_step), desc='Searching for best threshold'):
    p, r, f1, metrics = evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, all_gold_mentions, all_predicted_clusters, threshold)
    if f1 > best[-1]:
        best = p,r,f1
        best_metrics = metrics
        best_threshold = threshold

    return best + (best_metrics,)

def evaluate_by_threshold(all_cluster_logits, all_coref_logits, all_mention_logits, all_gold_clusters, all_gold_mentions, all_predicted_clusters, threshold):
    evaluator = CorefEvaluator()
    metrics = [0] * 5
    for i, (cluster_logits, coref_logits, gold_clusters, gold_mentions, predicted_clusters) in enumerate(
            zip(all_cluster_logits, all_coref_logits, all_gold_clusters, all_gold_mentions, all_predicted_clusters)):
        # prec_correct_mentions, prec_gold, prec_junk, prec_correct_gold_clusters, prec_correct_predict_clusters = \
        #     get_more_metrics(predicted_clusters, gold_clusters, gold_mentions)  #TODO: predicted_clusters[0]?
        # metrics[0] += prec_correct_mentions / len(all_cluster_logits)
        # metrics[1] += prec_gold / len(all_cluster_logits)
        # metrics[2] += prec_junk / len(all_cluster_logits)
        # metrics[3] += prec_correct_gold_clusters / len(all_cluster_logits)
        # metrics[4] += prec_correct_predict_clusters / len(all_cluster_logits)
        evaluator.update(predicted_clusters, [gold_clusters])
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


def pad_input_ids_and_mask_to_max_sentences(input_ids, mask, speaker_ids, input_ids_pads, mask_pads, speaker_ids_pads, max_sentences):
    if input_ids.shape[0] == max_sentences:
        return input_ids.unsqueeze(0), mask.unsqueeze(0), speaker_ids.unsqueeze(0)
    padded_input_ids = torch.cat([input_ids, input_ids_pads.repeat(max_sentences - input_ids.shape[0], 1)]).unsqueeze(0)
    padded_mask = torch.cat([mask, mask_pads.repeat(max_sentences - input_ids.shape[0], 1)]).unsqueeze(0)
    padded_speaker_ids = torch.cat([speaker_ids, speaker_ids_pads.repeat(max_sentences - input_ids.shape[0], 1, 1)]).unsqueeze(0)
    return padded_input_ids, padded_mask, padded_speaker_ids

def pad_mentions(gold_mentions, max_mentions):
    padded_gold_mentions = torch.tensor(np.asarray(gold_mentions + (max_mentions-len(gold_mentions)) * [(-1, -1)])).unsqueeze(0)
    return padded_gold_mentions

def tensor_and_remove_empty(batch, gold_mentions, args, input_ids_pads, mask_pads, speaker_ids_pads):
    input_ids, input_mask, sum_text_len, num_mentions, new_gold_mentions, speaker_ids, genre = [], [], [], [], [], [], []
    max_mentions = max([len(gm) for gm in gold_mentions])
    max_sentences = max([ii.shape[0] for ii in batch['input_ids']])
    num_examples = len(gold_mentions)
    for i in range(num_examples):
        padded_input_ids, padded_mask, padded_speaker_ids = pad_input_ids_and_mask_to_max_sentences(\
            torch.tensor(batch['input_ids'][i]).to(args.device), 
            torch.tensor(batch['input_mask'][i]).to(args.device), 
            torch.tensor(batch['speaker_ids'][i]).to(args.device),
            input_ids_pads, mask_pads, speaker_ids_pads, max_sentences)
        input_ids.append(padded_input_ids)
        input_mask.append(padded_mask)
        speaker_ids.append(padded_speaker_ids)
        genre.append(torch.tensor([batch['genre'][i]]).to(args.device))
        sum_text_len.append(torch.tensor([sum(batch['text_len'][i])]).to(args.device))
        new_gold_mentions.append(pad_mentions(gold_mentions[i], max_mentions))
        num_mentions.append(torch.tensor([len(gold_mentions[i])]).to(args.device))
    return torch.cat(input_ids).reshape(num_examples, max_sentences, -1), \
        torch.cat(input_mask).reshape(num_examples, max_sentences, -1), \
            torch.cat(sum_text_len).reshape(num_examples), \
                torch.cat(new_gold_mentions).reshape(num_examples, max_mentions, 2),\
                    torch.cat(num_mentions).reshape(num_examples),\
                        torch.cat(speaker_ids).reshape(num_examples, max_sentences, args.max_segment_len, -1),\
                    torch.cat(genre).reshape(num_examples, -1)
