import torch
import random
import jsonlines
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from consts import TOKENS_PAD, SPEAKER_PAD, GENRES, SPEAKER_END_TOKEN, SPEAKER_START_TOKEN, TOKENIZER_FILTERS, TOKENIZER_MAPS, BERT_WINDOW_SIZE


class OntonotesDataset(Dataset):

    def __init__(self, filepath, is_training, batch_size, args) -> None:
        super().__init__()
        self.is_training = is_training
        self.args = args
        self.batch_size = batch_size
        if args.bert_model:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --bert_model"
            )

        cache_filename = filepath + ".pickle"
        if os.path.exists(cache_filename):
            with open(cache_filename, mode="rb") as cache_f:
                self.examples = pickle.load(cache_f)
        else:
            self.examples = self._tokenize_docs(filepath)
            with open(cache_filename, mode="wb") as cache_f:
                pickle.dump(self.examples, cache_f)
        if is_training and args.limit_trainset >= 0:
            self.examples = self.examples[:args.limit_trainset]
        # if args.use_gold_mentions:
        #     for i, e in reversed(list(enumerate(self.examples))):
        #         if len(e['clusters']) == 0:
        #             del self.examples[i]
        # self.subtoken_maps = {}
        self.avg_spans = sum(len(doc["head2span"]) for doc in self.examples) / len(self.examples)

        if is_training:
            for i, e in reversed(list(enumerate(self.examples))):
                if e['document_id'] == 'nw/wsj/20/wsj_2013_0':
                    self.examples.pop(i)
        # self.tensorized_examples = self.get_all_tensorized_examples()

    def _tokenize_docs(self, path: str):
        print(f"Tokenizing documents at {path}...", flush=True)
        out = []
        filter_func = TOKENIZER_FILTERS.get(self.args.bert_model,
                                            lambda _: True)
        token_map = TOKENIZER_MAPS.get(self.args.bert_model, {})
        with jsonlines.open(path, mode="r") as data_f:
            for doc in data_f:
                doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                                   for cluster in doc["span_clusters"]]
                word2subword = []
                subwords = []
                word_id = []
                for i, word in enumerate(doc["cased_words"]):
                    tokenized_word = (token_map[word]
                                      if word in token_map
                                      else self.tokenizer.tokenize(word))
                    tokenized_word = list(filter(filter_func, tokenized_word))
                    word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
                    subwords.extend(tokenized_word)
                    word_id.extend([i] * len(tokenized_word))
                doc["word2subword"] = word2subword
                doc["subwords"] = subwords
                doc["word_id"] = word_id
                out.append(doc)
        print("Tokenization OK", flush=True)
        return out

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        tensorized_example = self._tensorize_example(example)
        return tensorized_example

    def _tensorize_example(self, example):
        subwords_batches = self.get_subwords_batches(example)

        special_tokens = np.array([self.tokenizer.cls_token_id,
                                   self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id])
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        # subwords_batches_tensor = torch.tensor(subwords_batches,
        #                                        device=self.args.device,
        #                                        dtype=torch.long)
        # subword_mask_tensor = torch.tensor(subword_mask,
        #                                    device=self.args.device)

        # Obtain bert output for selected batches only
        attention_mask = (subwords_batches != self.tokenizer.pad_token_id)

        copied_example = example.copy()
        copied_example['attention_mask'] = attention_mask
        copied_example['subwords_batches'] = subwords_batches
        copied_example['subword_mask'] = subword_mask
        
        return copied_example

    def get_subwords_batches(self, example) -> np.ndarray:
        """
        Turns a list of subwords to a list of lists of subword indices
        of max length == batch_size (or shorter, as batch boundaries
        should match sentence boundaries). Each batch is enclosed in cls and sep
        special tokens.

        Returns:
            batches of bert tokens [n_batches, batch_size]
        """
        batch_size = BERT_WINDOW_SIZE[self.args.bert_model] - 2  # to save space for CLS and SEP

        subwords = example["subwords"]
        subwords_batches = []
        start, end = 0, 0

        while end < len(subwords):
            end = min(end + batch_size, len(subwords))

            # Move back till we hit a sentence end
            if end < len(subwords):
                sent_id = example["sent_id"][example["word_id"][end]]
                while end and example["sent_id"][example["word_id"][end - 1]] == sent_id:
                    end -= 1

            length = end - start
            batch = [self.tokenizer.cls_token] + subwords[start:end] + [self.tokenizer.sep_token]
            batch_ids = [-1] + list(range(start, end)) + [-1]

            # Padding to desired length
            # -1 means the token is a special token
            batch += [self.tokenizer.pad_token] * (batch_size - length)
            batch_ids += [-1] * (batch_size - length)

            subwords_batches.append([self.tokenizer.convert_tokens_to_ids(token)
                                    for token in batch])
            start += length

        return np.array(subwords_batches)

    def get_all_tensorized_examples(self):
        tensorized_examples = []
        for ind, example in enumerate(self.examples):
            tensorized_example = self.tensorize_example(example)
            tensorized_examples.append(tensorized_example)
            if ind%100 == 0:
                print(ind)
        return tensorized_examples

    def get_tokenized_words_and_new_indices(self, sentence, speakers, total_tokens, word_idx, is_first):
        tekenized_sentence = []
        speaker_per_token = []
        word_idx_to_end_token_idx = dict()
        word_idx_to_start_token_idx = dict()

        for i in range(len(sentence)):
            word = sentence[i]
            if not is_first or i > 0:
                word = ' ' + word
            speaker = speakers[i]
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            tekenized_sentence += token_ids
            word_idx_to_start_token_idx[word_idx] = total_tokens  # +1 for <s>
            total_tokens += len(token_ids)
            word_idx_to_end_token_idx[word_idx] = total_tokens-1  # old_seq_len + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)
            speaker_per_token += [speaker] * len(token_ids)
            word_idx += 1
        return tekenized_sentence, word_idx, speaker_per_token, word_idx_to_start_token_idx, word_idx_to_end_token_idx, total_tokens

    def get_sentence_with_speaker(self, sentence, new_sentence, speaker, last_speaker_per_token, word_idx, total_tokens):
        speaker_per_token = []
        word_idx_to_end_token_idx = dict()
        word_idx_to_start_token_idx = dict()
        for i in range(len(sentence)):
            word = sentence[i]            
            word = ' ' + word
            word_speaker = speaker[i]
            token_ids = self.tokenizer.tokenize(word)
            speaker_prefix_token_num = 0
            if word_speaker != last_speaker_per_token:
                speaker_prefix = [SPEAKER_START_TOKEN] + self.tokenizer.tokenize(" " + word_speaker, add_special_tokens=False) + [SPEAKER_END_TOKEN]
                speaker_prefix_token_num = len(speaker_prefix)
                token_ids = speaker_prefix + token_ids
                new_sentence = new_sentence + f'{SPEAKER_START_TOKEN} {word_speaker}{SPEAKER_END_TOKEN}'
            new_sentence += word
            word_idx_to_start_token_idx[word_idx] = total_tokens + 1 + speaker_prefix_token_num  # +1 for <s>
            total_tokens += len(token_ids)
            word_idx_to_end_token_idx[word_idx] = total_tokens  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)
            speaker_per_token += [word_speaker] * len(token_ids)
            word_idx += 1
            last_speaker_per_token = speaker_per_token[-1]
        return new_sentence, speaker_per_token, word_idx, total_tokens, word_idx_to_start_token_idx, word_idx_to_end_token_idx

    def tensorize_example(self, example, is_training):
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
            current_total_tokens = 0
            tmp_total_tokens = 0
            sentence = sentences[sent_idx]
            speaker = speakers[sent_idx]
            speaker_per_token = ['[SPL]']
            new_sentence = ''
            tmp_new_sentence, tmp_speaker_per_token, tmp_word_idx, tmp_total_tokens, tmp_word_idx_to_start_token_idx, tmp_word_idx_to_end_token_idx = \
                self.get_sentence_with_speaker(sentence, new_sentence, speaker, speaker_per_token[-1], word_idx, total_tokens)
            current_total_tokens += tmp_total_tokens - total_tokens
            while current_total_tokens + 2 < max_sentence_length and sent_idx < len(sentences):
                total_tokens = tmp_total_tokens
                speaker_per_token += tmp_speaker_per_token
                word_idx = tmp_word_idx
                new_sentence = tmp_new_sentence
                word_idx_to_start_token_idx.update(tmp_word_idx_to_start_token_idx)
                word_idx_to_end_token_idx.update(tmp_word_idx_to_end_token_idx)

                sent_idx += 1
                if sent_idx >= len(sentences):
                    break
                
                sentence = sentences[sent_idx]
                speaker = speakers[sent_idx]
                tmp_new_sentence, tmp_speaker_per_token, tmp_word_idx, tmp_total_tokens, tmp_word_idx_to_start_token_idx, tmp_word_idx_to_end_token_idx = \
                    self.get_sentence_with_speaker(sentence, new_sentence, speaker, speaker_per_token[-1], word_idx, total_tokens)
                current_total_tokens += tmp_total_tokens - total_tokens
            if new_sentence == '':
                total_tokens = tmp_total_tokens
                speaker_per_token += tmp_speaker_per_token
                word_idx = tmp_word_idx
                new_sentence = tmp_new_sentence
                word_idx_to_start_token_idx.update(tmp_word_idx_to_start_token_idx)
                word_idx_to_end_token_idx.update(tmp_word_idx_to_end_token_idx)
                sent_idx += 1

            sent_input_ids = self.tokenizer.encode(new_sentence)

            speaker_per_token += ['[SPL]']

            total_tokens += 2

            cur_text_len = len(sent_input_ids)
            text_len.append(cur_text_len)
            
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
        # assert total_tokens == np.sum(input_mask), (total_tokens, np.sum(input_mask))

        doc_key = example["doc_key"][:2]
        genre = GENRES.get(doc_key, 0)

        genre = self.encode_genre_binary(genre)

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

        if is_training and len(input_ids) > self.args.max_training_sentences:
            tensorized_example = self.truncate_example(tensorized_example)

        tensorized_example['speaker_ids'] = self.encode_speaker_binary(tensorized_example['speaker_ids'])

        # calc clusters after truncation
        if len(tensorized_example['cluster_ids']) == 0:
            clusters = []
        else:
            cluster_ids_int = tensorized_example['cluster_ids'].astype(np.int)
            clusters = [[] for _ in range(cluster_ids_int.max())]
            for start, end, cluster_id in zip(tensorized_example['gold_starts'], tensorized_example['gold_ends'], cluster_ids_int):
                clusters[cluster_id-1].append((start, end))
            clusters = [c for c in clusters if len(c) > 0]
        
        tensorized_example['clusters'] = clusters

        return tensorized_example

    def encode_genre_binary(self, genre):
        encoded = np.zeros(len(GENRES)+1, dtype='uint8')
        encoded[genre] = 1
        return encoded

    def encode_speaker_binary(self, speaker_ids):
        speaker_ids_onehot = []
        for i in range(len(speaker_ids)):
            speaker_ids_onehot.append(np.eye(self.args.max_num_speakers, dtype='uint8')[speaker_ids[i]])
        return speaker_ids_onehot

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


