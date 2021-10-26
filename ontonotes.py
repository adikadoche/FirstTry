import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from consts import TOKENS_PAD, SPEAKER_PAD


class OntonotesDataset(Dataset):

    def __init__(self, filepath, is_training, batch_size, args) -> None:
        super().__init__()
        with open(filepath) as f:
            lines = f.readlines()
            self.examples = [json.loads(jsonline) for jsonline in lines]
        if args.limit_trainset >= 0:
            self.examples = self.examples[:args.limit_trainset]
        self.is_training = is_training
        self.args = args
        self.batch_size = batch_size
        self.subtoken_maps = {}
        self.genres = {g: i for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}
        if args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )
        self.tensorized_examples_training, self.tensorized_examples_eval = self.get_all_tensorized_examples()


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.tensorized_examples_training[index] if self.is_training else self.tensorized_examples_eval[index]

    def get_all_tensorized_examples(self):
        tensorized_training, tensorized_evel = [], []
        for example in self.examples:
            example_train, example_eval = self.tensorize_example(example)
            tensorized_training.append(example_train)
            tensorized_evel.append(example_eval)
        return tensorized_training, tensorized_evel

    def tensorize_example(self, example):
        tensorized_example = {}
        old_clusters = example["clusters"]
        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        # assert num_words == len(speakers), (num_words, len(speakers))
        speaker_dict = self.get_speaker_dict(self.flatten(speakers))
        sentence_map = [] #example['sentence_map']




        max_sentence_length = self.args.max_segment_len

        input_ids, input_mask, speaker_ids, text_len = [], [], [], []
        word_idx = 0
        total_tokens = 0
        word_idx_to_end_token_idx = dict()
        word_idx_to_start_token_idx = dict()
        sent_idx = 0
        while sent_idx < len(sentences):
            concat_sentence = []
            concat_speaker = []
            sentence = sentences[sent_idx]
            speaker = speakers[sent_idx]
            current_encoded = self.tokenizer.encode(' '.join(sentence))
            total_sent_encoded = 1
            current_len_encoded = len(current_encoded)
            while current_len_encoded < max_sentence_length - total_sent_encoded and sent_idx < len(sentences):
                concat_sentence += sentence
                concat_speaker += speaker
                sent_idx += 1
                if sent_idx >= len(sentences):
                    break
                sentence = sentences[sent_idx]
                speaker = speakers[sent_idx]
                total_sent_encoded += 1
                current_encoded = self.tokenizer.encode(' '.join(sentence))
                current_len_encoded += len(current_encoded)
            sent_input_ids = self.tokenizer.encode(' '.join(concat_sentence))
            cur_text_len = len(sent_input_ids)
            text_len.append(cur_text_len)

            speaker_per_token = ['[SPL]']
            for i in range(len(concat_sentence)):
                word = concat_sentence[i]
                if i > 0:
                    word = ' ' + word
                speaker = concat_speaker[i]
                token_ids = self.tokenizer.tokenize(word)
                word_idx_to_start_token_idx[word_idx] = total_tokens + 1  # +1 for <s>
                total_tokens += len(token_ids)
                word_idx_to_end_token_idx[word_idx] = total_tokens  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)
                speaker_per_token += [speaker] * len(token_ids)
                word_idx += 1
            speaker_per_token += ['[SPL]']







        # pronoun_ids = []
        # pronoun_indices = example.get('pronoun_indices', [])
        max_sentence_length = self.args.max_segment_len

        input_ids, input_mask, speaker_ids, text_len = [], [], [], []
        word_idx = 0
        total_tokens = 0
        word_idx_to_end_token_idx = dict()
        word_idx_to_start_token_idx = dict()
        sent_idx = 0
        while sent_idx < len(sentences):
            concat_sentence = []
            concat_speaker = []
            sentence = sentences[sent_idx]
            speaker = speakers[sent_idx]
            current_encoded = self.tokenizer.encode(' '.join(sentence))
            total_sent_encoded = 1
            current_len_encoded = len(current_encoded)
            while current_len_encoded < max_sentence_length - total_sent_encoded and sent_idx < len(sentences):
                concat_sentence += sentence
                concat_speaker += speaker
                sent_idx += 1
                if sent_idx >= len(sentences):
                    break
                sentence = sentences[sent_idx]
                speaker = speakers[sent_idx]
                total_sent_encoded += 1
                current_encoded = self.tokenizer.encode(' '.join(sentence))
                current_len_encoded += len(current_encoded)
            sent_input_ids = self.tokenizer.encode(' '.join(concat_sentence))
            cur_text_len = len(sent_input_ids)
            text_len.append(cur_text_len)

            speaker_per_token = ['[SPL]']
            for i in range(len(concat_sentence)):
                word = concat_sentence[i]
                if i > 0:
                    word = ' ' + word
                speaker = concat_speaker[i]
                token_ids = self.tokenizer.tokenize(word)
                word_idx_to_start_token_idx[word_idx] = total_tokens + 1  # +1 for <s>
                total_tokens += len(token_ids)
                word_idx_to_end_token_idx[word_idx] = total_tokens  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)
                speaker_per_token += [speaker] * len(token_ids)
                word_idx += 1
            speaker_per_token += ['[SPL]']

            total_tokens += 2
            
            sent_input_mask = [1] * cur_text_len
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker_per_token]
            sent_input_mask += [0] * (max_sentence_length - cur_text_len)
            sent_speaker_ids += [SPEAKER_PAD] * (max_sentence_length - cur_text_len)
            sent_input_ids += [TOKENS_PAD] * (max_sentence_length - cur_text_len)
            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
        clusters = [
            [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
            cluster in old_clusters]


        gold_mentions = sorted(tuple(m) for m in self.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1


        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert total_tokens == np.sum(input_mask), (total_tokens, np.sum(input_mask))

        doc_key = example["doc_key"]
        # self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        genre = self.genres.get(doc_key[:2], 0)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        tensorized_example['input_ids'] = input_ids
        tensorized_example['input_mask'] = input_mask
        tensorized_example['text_len'] = text_len
        tensorized_example['speaker_ids'] = speaker_ids
        tensorized_example['genre'] = genre
        tensorized_example['gold_starts'] = gold_starts
        tensorized_example['gold_ends'] = gold_ends
        tensorized_example['cluster_ids'] = cluster_ids
        tensorized_example['sentence_map'] = sentence_map
        tensorized_example_eval = tensorized_example

        if len(input_ids) > self.args.max_training_sentences:
            tensorized_example_eval = self.truncate_example(tensorized_example)

        # calc clusters after truncation
        tensorized_example['clusters'] = self.calc_clusters(tensorized_example)
        tensorized_example_eval['clusters'] = self.calc_clusters(tensorized_example_eval)

        return tensorized_example, tensorized_example_eval

    def calc_clusters(self, tensorized_example):
        if len(tensorized_example['cluster_ids']) == 0:
            clusters = []
        else:
            cluster_ids_int = tensorized_example['cluster_ids'].astype(np.int)
            clusters = [[] for _ in range(cluster_ids_int.max())]
            for start, end, cluster_id in zip(tensorized_example['gold_starts'], tensorized_example['gold_ends'], cluster_ids_int):
                clusters[cluster_id-1].append((start, end))
            clusters = [c for c in clusters if len(c) > 0]
        return clusters

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def truncate_example(self, tensorized_example, sentence_offset=None):
        max_training_sentences = self.args.max_training_sentences
        num_sentences = tensorized_example['input_ids'].shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0,
                                         num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = sum(tensorized_example['text_len'][:sentence_offset])
        num_words = sum(tensorized_example['text_len'][sentence_offset:sentence_offset + max_training_sentences])
        tensorized_example['input_ids'] = tensorized_example['input_ids'][sentence_offset:sentence_offset + max_training_sentences, :]
        tensorized_example['input_mask'] = tensorized_example['input_mask'][sentence_offset:sentence_offset + max_training_sentences, :]
        tensorized_example['speaker_ids'] = tensorized_example['speaker_ids'][sentence_offset:sentence_offset + max_training_sentences, :]
        tensorized_example['text_len'] = tensorized_example['text_len'][sentence_offset:sentence_offset + max_training_sentences]

        # sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(tensorized_example['gold_ends'] >= word_offset, tensorized_example['gold_starts'] < word_offset + num_words)
        tensorized_example['gold_starts'] = tensorized_example['gold_starts'][gold_spans] - word_offset
        tensorized_example['gold_ends'] = tensorized_example['gold_ends'][gold_spans] - word_offset
        tensorized_example['cluster_ids'] = tensorized_example['cluster_ids'][gold_spans]

        return tensorized_example

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': SPEAKER_PAD, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.args.max_num_speakers:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict


