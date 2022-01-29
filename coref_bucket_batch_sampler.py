import logging
from typing import List, Iterable
import random
import math
import torch

from torch.utils import data
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BucketBatchSampler(DataLoader):
    def __init__(
            self,
            data_source: data.Dataset,
            max_total_seq_len: int,
            sorting_keys: List[str] = None,
            padding_noise: float = 0.1,
            drop_last: bool = False,
            batch_size_1: bool = False,
            n_gpu: int = 1
    ):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.max_total_seq_len = max_total_seq_len
        self.n_gpu = n_gpu
        self.data_source = data_source
        data_source.examples.sort(key=lambda x: len(x[1].token_ids), reverse=True)
        self.drop_last = drop_last
        self.batches = self.prepare_batches() if not batch_size_1 else self.prepare_eval_batches()

    def prepare_gpu_batch(self, gpu_batch):
        max_length = len(gpu_batch[0][0].token_ids)
        max_mentions = 0
        for i in range(len(gpu_batch)):
            gpu_batch[i] = self.data_source.pad_batch(gpu_batch[i], max_length)
            max_mentions = max(max_mentions, max([torch.sum(gpu_batch[i][-2][j]) for j in range(gpu_batch[i][-2].shape[0])]))
        gpu_batch = tuple(torch.stack([example[i] for example in gpu_batch], dim=0) for i in [0,1]) + \
                (torch.stack([example[2] for example in gpu_batch], dim=0)[:,:,:int(max_mentions),:], \
                torch.stack([example[3] for example in gpu_batch], dim=0)[:,:,:int(max_mentions)],
                [gpu_batch[i][-1] for i in range(len(gpu_batch))],)
        return gpu_batch
  
    def prepare_batches(self):
        batches = []
        batch = []
        gpu_batch = []
        per_example_batch_len = 0
        for _, elem in self.data_source:  #TODO: add n_gpu dim
            if len(batch) == 0:
                # TODO change to config.attention_window
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            elif (len(batch) + 1) * per_example_batch_len > self.max_total_seq_len:
                if len(gpu_batch) == 0 or (len(gpu_batch) < self.n_gpu and len(batch) == len(gpu_batch[-1])):
                    gpu_batch.append(batch)
                else:
                    batches.append(self.prepare_gpu_batch(gpu_batch))
                    gpu_batch = [batch]
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            batch.append(elem)
        if len(gpu_batch) == 0 or (len(gpu_batch) < self.n_gpu and len(batch) == len(gpu_batch[-1])):
            gpu_batch.append(batch)
        else:
            batches.append(self.prepare_gpu_batch(gpu_batch))
            gpu_batch = [batch]
        batches.append(self.prepare_gpu_batch(gpu_batch))

        return batches

    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil((example_len + 2) / 512) * 512

    def prepare_eval_batches(self):  
        batches = []
        for doc_key, elem in self.data_source:
            batches.append((doc_key, self.prepare_gpu_batch([[elem]])))
        return batches
