import random
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

TRAIN_RAW_PATH = '/home/gamir/adiz/datasets/toynames/GirlsTrain.txt'
DEV_RAW_PATH = '/home/gamir/adiz/datasets/toynames/BoysDev.txt'

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-large-4096', cache_dir='/home/gamir/adiz/Code/runs/s2e-coref/cache_dir/')


def make_sentences(raw_lines, num_of_texts):
    lines = []
    clusters = []
    for i in range(num_of_texts):
        text_len = random.randint(20, 2500)
        num_clusters = random.randint(1, 100)
        cur_names = random.sample(raw_lines, num_clusters)
        name_token_len = [len(tokenizer.tokenize(r)) for r in cur_names]
        sequence_distribution = [random.uniform(0, 1) for _ in range(num_clusters)]
        line_sequence_list = random.choices(cur_names, weights=sequence_distribution, k=text_len)

        for ii, n in enumerate(cur_names):
            c = line_sequence_list.count(n)
            if c == 1:
                cur_names.remove(n)
                name_token_len.pop(ii)
                line_sequence_list.remove(n)
            elif c == 0:
                cur_names.remove(n)
                name_token_len.pop(ii)

        cur_clusters = [[] for _ in range(len(cur_names))]
        cluster_dict = {c:cur_names.index(c) for c in cur_names}
        ind = 1
        for j in range(len(line_sequence_list)):
            c = cluster_dict[line_sequence_list[j]]
            cur_clusters[c].append((ind, ind+name_token_len[c]-1))
            ind += name_token_len[c]

        lines.append(' '.join(line_sequence_list))
        clusters.append(cur_clusters)
    return lines, clusters

def prepare_batches(text, clusters):
    batches = []
    for i in range(len(text)):
        batch = dict()
        batch['input_ids'] = [tokenizer.encode(text[i])]
        batch['clusters'] = clusters[i]
        for c in clusters[i]:
            for m in c:
                if m[1] > len(batch['input_ids'][0]):
                    print('x')
        batch['text_len'] = [len(batch['input_ids'][0])]
        batch['input_mask'] = [[1] * batch['text_len'][0]]
        batches.append(batch)
    return batches

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

def get_batches(is_training, num_of_texts):
    path = f'/home/gamir/adiz/datasets/toynames/{is_training}_{str(num_of_texts)}.txt'
    if os.path.isfile(path):
        batches = read_data_file(path)
        return batches
    else:
        path_raw = TRAIN_RAW_PATH if is_training else DEV_RAW_PATH
        with open(path_raw) as f:
            raw = f.readlines()
            raw = [r.strip() for r in raw]
        batches = prepare_batches(*make_sentences(raw, num_of_texts))
        write_data_file(path, batches)
        return batches


class ToyDataset(Dataset):
    def __init__(self, is_training, num_of_texts) -> None:
        super().__init__()
        self.examples = get_batches(is_training, num_of_texts)

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

def get_toy_data_objects(is_training, num_of_texts, args):
    dataset = ToyDataset(is_training, num_of_texts)
    loader = DataLoader(dataset, batch_size=1,
                             pin_memory=not args.no_cuda, num_workers=args.num_workers, collate_fn=collate_fn,
                             worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 2**32))

    if is_training:
        per_gpu_batch_size = args.per_gpu_train_batch_size
    else:
        per_gpu_batch_size = args.per_gpu_eval_batch_size

    batch_size = per_gpu_batch_size * max(1, args.n_gpu)
    return dataset, loader, batch_size
