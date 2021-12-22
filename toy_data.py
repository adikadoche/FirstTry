from collections import namedtuple
import random
import os
import json
import torch
import re
import numpy as np

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from consts import LETTERS_LIST

FUNCTIONS_NAMES = ['letters', 'structural', 'sequences']
TOKENIZER = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir="/home/gamir/adiz/Code/runs/firsttry/cache_dir/")
TOY_DATA_PATH = '/home/gamir/adiz/datasets/ontonotes/toy_data/'

def create_letters_dataset(is_add_junk=False, num_of_texts = 3000):
    if is_add_junk:
        letters_list = random.sample(LETTERS_LIST, 20)
        junk_letters_list = [l for l in LETTERS_LIST if l not in letters_list]
    else:
        letters_list = LETTERS_LIST
        junk_letters_list = []
    text = []
    clusters = []
    for _ in range(num_of_texts):
        text_len = random.randint(40, 3500)
        bkgd_text_len = int(random.uniform(0.6, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.randint(1, len(letters_list)-5)
        cur_letters = random.sample(letters_list, num_clusters)
        if len(junk_letters_list) > 0:
            bgkd_letters = junk_letters_list
        else:
            bgkd_letters = [l for l in letters_list if l not in cur_letters]
        sequence_distribution = [random.uniform(0, 1) for _ in range(num_clusters)]
        line_sequence_list = random.choices(cur_letters, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(bgkd_letters))]
        line_bkgd_list = random.choices(bgkd_letters, weights=bgkd_distribution, k=bkgd_text_len)

        sorted_pairs = [(x, y) for x, y in sorted(zip(line_sequence_list, sequence_indices), key=lambda pair: pair[1])]

        line_list = line_bkgd_list

        for letter, index in reversed(sorted_pairs):
            line_list.insert(index, letter)        

        text.append(' '.join(line_list))

        letters_indices = {}
        text_cluster = []
        for i, letter in enumerate(line_list):
            if letter in cur_letters:
                if letter not in letters_indices.keys():
                    letters_indices[letter] = len(letters_indices)
                    text_cluster.append([])
                text_cluster[letters_indices[letter]].append([i+1, i+1])
        
        clusters.append(text_cluster)

    return text, clusters

def create_sequences_dataset(is_add_junk=False, num_of_texts = 3000):
    SEQUENCES = []
    for _ in range(5):
        seq_len = random.randint(2, 5)
        seq = random.choices(LETTERS_LIST, k=seq_len)
        if seq not in SEQUENCES:
            is_contained = False
            for s in SEQUENCES:
                if ' '.join(s) in ' '.join(seq) or ' '.join(seq) in ' '.join(s):
                    is_contained = True
                    break
            if not is_contained:
                SEQUENCES.append(seq)
    text = []
    clusters = []
    for t in range(num_of_texts):
        clusters.append([])
        text_len = random.randint(40, 3500)
        bkgd_text_len = int(random.uniform(0.6, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.choices(list(range(1, min(int(text_len/4), len(SEQUENCES)))), k=1, weights=reversed(list(range(1, min(int(text_len/4), len(SEQUENCES))))))
        cur_sequenceds = random.sample(SEQUENCES, num_clusters[0])
        sequence_distribution = [random.uniform(0, 1) for _ in range(len(cur_sequenceds))]
        sequence_text_len = int(sequence_text_len / (sum([len(cur_sequenceds[i]) * sequence_distribution[i] for i in range(len(cur_sequenceds))]) / sum(sequence_distribution)))
        line_sequence_list = random.choices(cur_sequenceds, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(LETTERS_LIST))]
        line_bkgd_list = random.choices(LETTERS_LIST, weights=bgkd_distribution, k=bkgd_text_len)

        sorted_pairs = [(x, y) for x, y in sorted(zip(line_sequence_list, sequence_indices), key=lambda pair: pair[1])]

        line_list = line_bkgd_list

        for i, (sequence, index) in enumerate(reversed(sorted_pairs)):
            i = len(sorted_pairs)-1 - i
            if i < len(sorted_pairs)-1 and sorted_pairs[i][0] == sorted_pairs[i+1][0] and sorted_pairs[i+1][1]-sorted_pairs[i][1] == 1:
                continue
            line_list[index:index] = sequence

        is_non_singelton = True
        while is_non_singelton:
            line_text = ' '.join(line_list)
            is_non_singelton = False
            for s in SEQUENCES:
                c = line_text.count(' '.join(s))
                if c == 1:
                    is_non_singelton = True
                    line_text = line_text.replace(' '.join(s), '')
            line_list = re.split(' +', line_text)
        line_list = [l for l in line_list if l in LETTERS_LIST]
        for s in SEQUENCES:
            sequence_cluster = []
            for j in range(len(line_list) - len(s) + 1):
                if line_list[j:j+len(s)] == s:
                    sequence_cluster.append([j+1, j+len(s)-1+1])
            if len(sequence_cluster) > 0:
                clusters[t].append(sequence_cluster)

        text.append(' '.join(line_list))

    return text, clusters

def create_structural_dataset(is_add_junk=False, num_of_texts = 3000):
    OneSideSequencePattern = namedtuple("OneSideSequencePattern", ["index_delta", "sequence"])
    SequencePattern = namedtuple("SequencePattern", ["left", "right", "total_len"])
    SEQUENCES = []
    for _ in range(70):
        left_right_both = random.randint(0, 2)
        if left_right_both == 0:
            sequence_pattern_list = [True, True]
        elif left_right_both == 1:
            sequence_pattern_list = [True, False]
        else:
            sequence_pattern_list = [False, True]
        
        total_len = 0
        for i in range(2):
            if not sequence_pattern_list[i]:
                sequence_pattern_list[i] = OneSideSequencePattern(index_delta=0, sequence='')
            else:
                sequence_pattern_list[i] = OneSideSequencePattern(index_delta=random.randint(1, 4), \
                                                                sequence=random.choices(LETTERS_LIST, k=random.randint(1, 3)))
                total_len += sequence_pattern_list[i].index_delta + len(sequence_pattern_list[i].sequence)
        cur_seq_pattern = SequencePattern(left=sequence_pattern_list[0], right=sequence_pattern_list[1], total_len=total_len+1)
        if cur_seq_pattern not in SEQUENCES:
            SEQUENCES.append(cur_seq_pattern)

    text = []
    clusters = []
    for t in range(num_of_texts):
        clusters.append([])
        text_len = random.randint(40, 3500)
        bkgd_text_len = int(random.uniform(0.6, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.choices(list(range(1, min(int(text_len/4), len(SEQUENCES)))), k=1, weights=reversed(list(range(1, min(int(text_len/4), len(SEQUENCES))))))
        cur_sequenceds = random.sample(SEQUENCES, num_clusters[0])
        sequence_distribution = [random.uniform(0, 1) for _ in range(len(cur_sequenceds))]
        sequence_text_len = int(sequence_text_len / (sum([cur_sequenceds[i].total_len * sequence_distribution[i] for i in range(len(cur_sequenceds))]) / sum(sequence_distribution)))
        line_sequence_list = random.choices(cur_sequenceds, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(LETTERS_LIST))]
        line_bkgd_list = random.choices(LETTERS_LIST, weights=bgkd_distribution, k=bkgd_text_len)

        sorted_pairs = [(x, y) for x, y in sorted(zip(line_sequence_list, sequence_indices), key=lambda pair: pair[1])]

        line_list = line_bkgd_list

        for sequence, index in reversed(sorted_pairs):
            if index+sequence.right.index_delta >= len(line_bkgd_list) or index-sequence.left.index_delta < 0:
                continue
            part_of_text = line_bkgd_list[index-sequence.left.index_delta:index+sequence.right.index_delta+1]
            cur_sequence = []
            cur_sequence += sequence.left.sequence if sequence.left.sequence != '' else []
            cur_sequence += part_of_text
            cur_sequence += sequence.right.sequence if sequence.right.sequence != '' else []
            line_list[index:index] = cur_sequence

        text.append(' '.join(line_list))
        for s in cur_sequenceds:
            s_cluster = []
            for i in range(len(line_list) - s.total_len + 1):
                if line_list[i:i+len(s.left.sequence)] == s.left.sequence and \
                    line_list[i+len(s.left.sequence)+s.left.index_delta+1+s.right.index_delta:i+s.total_len] == s.right.sequence:
                    s_cluster.append([i+len(s.left.sequence)+s.left.index_delta+1, i+len(s.left.sequence)+s.left.index_delta+1])
            if len(s_cluster) > 0:
                clusters[-1].append(s_cluster)
    return text, clusters

def prepare_batches(text, clusters, factor):
    batches_dict = {}
    batches = []
    for i in range(int(factor*len(text))):
        batch = dict()
        batch['clusters'] = clusters[i]
        batch['input_ids'] = [TOKENIZER.encode(text[i])]
        batch['text_len'] = [len(batch['input_ids'][0])]
        batch['input_mask'] = [[1] * batch['text_len'][0]]
        batches.append(batch)
    batches_dict[int(factor*len(text))] = batches
    batches = []
    for i in range(int(factor*len(text)), len(text)):
        batch = dict()
        batch['clusters'] = clusters[i]
        batch['input_ids'] = [TOKENIZER.encode(text[i])]
        batch['text_len'] = [len(batch['input_ids'][0])]
        batch['input_mask'] = [[1] * batch['text_len'][0]]
        batches.append(batch)
    batches_dict[len(text)-int(factor*len(text))] = batches
    return batches_dict

def read_data_file(path):
    with open(path, "r") as reader:
        lines = reader.readlines()
        batches = [json.loads(jsonline) for jsonline in lines]
    return batches

def write_data_file(path, batches):
    with open(path, "w") as writer:
        for i in range(len(batches)):
            writer.write(json.dumps(batches[i]))
            writer.write('\n')

FUNCS = {FUNCTIONS_NAMES[0]: create_letters_dataset, FUNCTIONS_NAMES[1]: create_structural_dataset, FUNCTIONS_NAMES[2]: create_sequences_dataset}
NUM_OF_TEXTS = 20

def get_batches(type, is_training, is_add_junk, factor, num_of_texts=NUM_OF_TEXTS):
    path = TOY_DATA_PATH + f'{type}_{int(factor*num_of_texts)}_{is_training}'
    if type == FUNCTIONS_NAMES[0] and is_add_junk:
        path += '_junk'
    path += '.txt'
    if os.path.isfile(path):
        batches = read_data_file(path)
        return batches
    else:
        batches = prepare_batches(*FUNCS[type](is_add_junk, num_of_texts), factor)
        write_data_file(path, batches[int(factor*num_of_texts)])
        path = TOY_DATA_PATH + f'{type}_{num_of_texts-int(factor*num_of_texts)}_{not is_training}'
        if type == FUNCTIONS_NAMES[0] and is_add_junk:
            path += '_junk'
        path += '.txt'
        write_data_file(path, batches[num_of_texts-int(factor*num_of_texts)])
        return batches[int(factor*num_of_texts)]

# get_batches(FUNCTIONS_NAMES[0], 100)

class ToyDataset(Dataset):

    def __init__(self, type, is_training, is_add_junk=False, factor=0.8, num_of_texts=NUM_OF_TEXTS) -> None:
        super().__init__()
        self.examples = get_batches(type, is_training, is_add_junk, factor, num_of_texts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        example['input_ids'] = np.array(example['input_ids'])
        return example

def collate_fn(batch):
    if batch[0] == []:
        return []
    batch_concat = {}
    for key in batch[0].keys():
        batch_concat[key] = [0] * len(batch)
        for i in range(len(batch)):
            batch_concat[key][i] = batch[i][key]
    return batch_concat

def get_toy_data_objects(type, is_training, args, factor, num_of_texts=NUM_OF_TEXTS):
    dataset = ToyDataset(type, is_training, args.add_junk or not args.use_gold_mentions, factor, num_of_texts)
    loader = DataLoader(dataset, batch_size=1,
                             pin_memory=not args.no_cuda, num_workers=args.num_workers, collate_fn=collate_fn,
                             worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 2**32))

    return dataset, loader
