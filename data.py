import json
import logging
import os
import pickle
from collections import namedtuple
import numpy as np

import torch

from consts import SPEAKER_START_ID, SPEAKER_END_ID, NULL_ID_FOR_COREF
from torch.utils.data import Dataset, RandomSampler, DistributedSampler, SequentialSampler, DataLoader
from ontonotes import OntonotesDataset


CorefExample = namedtuple("CorefExample", ["words", "token_ids", "clusters"])

logger = logging.getLogger(__name__)


class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=-1):
        self.tokenizer = tokenizer
        logger.info(f"Reading dataset from {file_path}")
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(file_path)
        self.max_seq_length = max_seq_length
        self.examples, self.lengths, self.num_examples_filtered = self._tokenize(examples)
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length.")

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _tokenize(self, examples):
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = [0]  # for <s>

            token_ids = []
            words_with_speaker = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = [SPEAKER_START_ID] + self.tokenizer.tokenize(" " + speaker,
                                                                             add_special_tokens=False) + [SPEAKER_END_ID]
                    words_with_speaker += [SPEAKER_START_ID, speaker, SPEAKER_END_ID]
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.tokenize(word, add_special_tokens=False)
                words_with_speaker.append(word)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue

            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            lengths.append(len(token_ids))

            coref_examples.append(((doc_key, end_token_idx_to_word_idx), CorefExample(words=words_with_speaker, token_ids=token_ids, clusters=new_clusters)))
        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    # def pad_clusters_inside(self, clusters):
    #     return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
    #             in clusters]

    # def pad_clusters_outside(self, clusters):
    #     return clusters + [[]] * (self.max_num_clusters - len(clusters))  #wasety?

    # def pad_clusters(self, clusters):
    #     clusters = self.pad_clusters_outside(clusters)
    #     clusters = self.pad_clusters_inside(clusters)
    #     return clusters

    def pad_mentions(self, mentions):
        return mentions + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_mention_num - len(mentions)), \
            torch.cat([torch.ones(len(mentions)), torch.zeros(self.max_mention_num - len(mentions))], 0)  #wasety?

    def pad_batch(self, batch, max_length):     #wasety?
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        clusters_list = []
        for example in batch:
            encoded_dict = self.tokenizer(text=example.words,
                                                      add_special_tokens=True,
                                                      is_split_into_words=True,
                                                      padding='max_length',
                                                      max_length=max_length,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')
            clusters_list.append(example.clusters)
            mentions, mentions_mask = self.pad_mentions(list(set(tuple(m) for gc in example.clusters for m in gc)))
            #         input_ids, input_mask, gold_clusters, gold_mentions, gold_mentions_mask = batch
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + \
                (torch.tensor(mentions), mentions_mask,)
            padded_batch.append(example)
        tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))
        return tensored_batch + (clusters_list,)


def get_dataset(args, tokenizer, evaluate=False):
    read_from_cache, file_path = False, ''
    predict_file_cache = f"{args.predict_file_cache}_{str(args.n_gpu)}.pkl"
    train_file_cache = f"{args.train_file_cache}_{str(args.n_gpu)}.pkl"
    if evaluate and os.path.exists(predict_file_cache):
        file_path = predict_file_cache
        read_from_cache = True
    elif (not evaluate) and os.path.exists(train_file_cache):
        file_path = train_file_cache
        read_from_cache = True

    if read_from_cache:
        logger.info(f"Reading dataset from {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    file_path, cache_path = (args.predict_file, predict_file_cache) if evaluate else \
        (args.train_file, train_file_cache)

    coref_dataset = CorefDataset(file_path, tokenizer, max_seq_length=args.max_seq_length)
    with open(cache_path, 'wb') as f:
        pickle.dump(coref_dataset, f)

    return coref_dataset

def collate_fn(batch):
    batch_concat = {}
    for key in batch[0].keys():
        batch_concat[key] = [0] * len(batch)
        for i in range(len(batch)):
            batch_concat[key][i] = batch[i][key]
    return batch_concat

def get_data_objects(args, data_file_path, is_training):
    if is_training:
        per_gpu_batch_size = args.per_gpu_train_batch_size
    else:
        per_gpu_batch_size = args.per_gpu_eval_batch_size

    batch_size = per_gpu_batch_size * max(1, args.n_gpu)
    dataset = OntonotesDataset(data_file_path, is_training, batch_size, args)
    # Note that DistributedSampler samples randomly
    if is_training:
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        logger.info("Loaded train data")
    else:
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        logger.info("Loaded eval data")

    # sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                             pin_memory=not args.no_cuda, collate_fn=collate_fn, num_workers=args.num_workers,
                             worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 2**32))

    return dataset, sampler, loader, batch_size

def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]
