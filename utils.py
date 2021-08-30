import json
import os
from datetime import datetime
from time import time
from typing import List
import git
import torch
import numpy as np
import torch.distributed as dist


from consts import NULL_ID_FOR_COREF


def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]


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


def write_meta_data(output_dir, args):
    output_path = os.path.join(output_dir, "meta.json")
    repo = git.Repo(search_parent_directories=True)
    hexsha = repo.head.commit.hexsha
    ts = time()
    print(f"Writing {output_path}")
    with open(output_path, mode='w') as f:
        json.dump(
            {
                'git_hexsha': hexsha,
                'args': {k: str(v) for k, v in args.__dict__.items()},
                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            },
            f,
            indent=4,
            sort_keys=True)
        print(file=f)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / d_model) #TODO: make sure returns float
    return pos * angle_rates


def build_positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


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
        for cluster_id, cluster in enumerate(gold_clusters):
            for mention in cluster:
                mention_index = gold_mentions.index(tuple(mention))
                assert mention_index >= 0
                gold_per_token[cluster_id, mention_index] = 1

    return gold_per_token

def calc_predicted_clusters(cluster_logits, coref_logits, threshold, gold_mentions: List):
    cluster_logits = cluster_logits.numpy() >= threshold
    coref_logits = coref_logits.numpy() >= threshold

    if gold_mentions is None:
        bsz, num_of_clusters, _ = coref_logits.shape
        clusters = []
        for i in range(0, num_of_clusters):
            current_cluster = []

            first_token_index = None
            for token_index, token_logit in enumerate(coref_logits[0, i]):
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
        clusters = []

        if coref_logits.shape[-1] > 0:
            mention_to_cluster_id = coref_logits.squeeze(0).argmax(0) # [mentions]
            cluster_ids = mention_to_cluster_id.unique()

            for cluster_id in cluster_ids:
                current_cluster = []
                for mention_id in (mention_to_cluster_id == cluster_id).nonzero():
                    current_cluster.append(gold_mentions[mention_id])
                clusters.append(current_cluster)

    return clusters


