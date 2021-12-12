# -*- coding: utf-8 -*-
"""coref_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1grjpicqh9EHuQf3Gqqj_REAB2RGIzK1x
"""

from collections import Counter
import numpy as np
from colorama import Back, Style
from pip._vendor.colorama import Fore

from utils import calc_predicted_clusters
from consts import PRONOUNS

FORES = [Fore.BLUE,
         Fore.CYAN,
         Fore.GREEN,
         Fore.MAGENTA,
         Fore.RED,
         Fore.YELLOW]
BACKS = [Back.LIGHTBLACK_EX,
         Back.LIGHTCYAN_EX,
         Back.LIGHTGREEN_EX,
         Back.LIGHTMAGENTA_EX,
         Back.LIGHTRED_EX,
         Back.LIGHTYELLOW_EX]
COLOR_WHEEL = FORES + [f + b for i, f in enumerate(FORES) for j, b in enumerate(BACKS) if i != j]


def coref_pprint(tokens, clusters, correct_flags=None, partial_cluster_to_color=None, complete_miss_flags=None, cluster_to_linked_entities=None):
    clusters = [tuple(tuple(m) for m in c) for c in clusters]

    if correct_flags is None:
        cluster_to_color = {c: COLOR_WHEEL[i % len(COLOR_WHEEL)] for i, c in enumerate(clusters)}
    elif partial_cluster_to_color is None:
        cluster_to_color = {c: COLOR_WHEEL[i % len(COLOR_WHEEL)] if not correct_flags[i] else Fore.LIGHTBLACK_EX for
                            i, c in enumerate(clusters)}
    else:
        cluster_to_color = {}
        unused_colors = list(COLOR_WHEEL)
        for c in partial_cluster_to_color.values():
            if c in unused_colors:
                unused_colors.remove(c)

        for i, c in enumerate(clusters):
            if correct_flags[i]:
                cluster_to_color[c] = Fore.LIGHTBLACK_EX
            elif c in partial_cluster_to_color:
                cluster_to_color[c] = partial_cluster_to_color[c]
            elif len(unused_colors) > 0:
                cluster_to_color[c] = unused_colors.pop()
            else:
                cluster_to_color[c] = COLOR_WHEEL[i % len(COLOR_WHEEL)]

    pretty_str = ''
    color_stack = []
    for i, t in enumerate(tokens):
        # find all mentions (from all clusters) that start at i
        # and sort all mentions/clusters by end
        mentions_start_at_i = []
        for c in clusters:
            for start, end in c:
                if start == i:
                    mentions_start_at_i.append((end, c))

        mentions_start_at_i = sorted(mentions_start_at_i, reverse=True)
        if mentions_start_at_i:
            for end, c in mentions_start_at_i:
                cluster_color = cluster_to_color[c]
                color_stack.append(cluster_color)

            _, c = mentions_start_at_i[-1]
            cluster_color = cluster_to_color[c]
            pretty_str += Style.RESET_ALL + Style.BRIGHT + cluster_color

        pretty_str += t + u' '

        # count how many mentions end at i
        mentions_end_at_i_count = len(list(_ for c in clusters for start, end in c if i == end))
        if mentions_end_at_i_count > 0:
            # pop them out of color stack
            color_stack = color_stack[:-mentions_end_at_i_count]
            pretty_str += Style.RESET_ALL
            if color_stack:
                pretty_str += Style.BRIGHT + color_stack[-1]

        # for c in clusters:
        #     for start, end in c:
        #         if i == end:
        #             pretty_str += Style.RESET_ALL
        #             color_stack.pop(-1)
        #             if color_stack:
        #                 pretty_str += Style.BRIGHT + color_stack[-1]

    pretty_str += Style.RESET_ALL + "\n"
    pretty_str = pretty_str.replace(' ##', '')

    # add cluster to the end
    for i, c in enumerate(clusters):
        cluster_color = cluster_to_color[c]
        mentions = [" ".join(tokens[start:end+1]).replace(' ##', '') for start, end in c]
        pretty_str += Style.BRIGHT + cluster_color + "[" + " | ".join(mentions) + "]" + Style.RESET_ALL

        if complete_miss_flags is not None and complete_miss_flags[i]:
            pretty_str += " - completely missed"

        if cluster_to_linked_entities:
            entities = cluster_to_linked_entities[i]
            if len(entities) > 0:
                pretty_str += " - {}".format(entities)

        pretty_str += "\n"

    print(pretty_str)
    return cluster_to_color


def flatten(l):
    return [item for sublist in l for item in sublist]


def match_clusters(gold, pred):
    sorted_gold = sorted([tuple(sorted(tuple(m) for m in c)) for c in gold])
    sorted_pred = sorted([tuple(sorted(tuple(m) for m in c)) for c in pred])

    gold_has_correct_pred = []
    pred_correct = [False] * len(sorted_pred)
    pred_to_most_similar_gold = [-1] * len(sorted_pred)
    pred_to_most_similar_golds_list = [[]] * len(sorted_pred)
    gold_to_most_similar_pred = [-1] * len(sorted_gold)
    gold_is_completely_missed = [False] * len(sorted_gold)

    for gold_index, gold_c in enumerate(sorted_gold):
        # find most similar cluster in pred
        most_similar_pred = None
        most_similar_pred_score = 0
        most_similar_pred_index = -1

        total_score = 0
        for i, pc in enumerate(sorted_pred):
            score = len(set(gold_c).intersection(set(pc)))
            total_score += score
            if score > most_similar_pred_score:
                most_similar_pred_score = score
                most_similar_pred = pc
                most_similar_pred_index = i

        if total_score == 0:
            gold_is_completely_missed[gold_index] = True

        is_correct = most_similar_pred == gold_c
        gold_has_correct_pred.append(is_correct)

        if most_similar_pred_index >= 0:
            pred_correct[most_similar_pred_index] |= is_correct
            gold_to_most_similar_pred[gold_index] = most_similar_pred_index
            pred_to_most_similar_golds_list[most_similar_pred_index].append(gold_index)

            if pred_to_most_similar_gold[most_similar_pred_index] == -1:
                pred_to_most_similar_gold[most_similar_pred_index] = gold_index

    return sorted_gold, gold_has_correct_pred,\
           sorted_pred, pred_correct, pred_to_most_similar_gold, pred_to_most_similar_golds_list,\
           gold_is_completely_missed, gold_to_most_similar_pred


def is_cluster_contains_linked_entities(cluster, entities_per_sentence, sentences, output_linked_entities=False):
    found_entity_in_cluster = False
    entities_found = set()
    for start, end in cluster:
        sentence_index = 0
        sentence = sentences[sentence_index]
        while start >= len(sentence):
            start -= len(sentence)
            end -= len(sentence)
            sentence_index += 1
            sentence = sentences[sentence_index]

        entities = entities_per_sentence[sentence_index]

        # calc mention in-sentences text location (without ##)
        char_start = len(' '.join(sentence[0:start]).replace(' ##', '').replace('[CLS] ', '')) + 1
        char_end = len(' '.join(sentence[0:end + 1]).replace(' ##', '').replace('[CLS] ', ''))

        text = ' '.join(sentence).replace(' ##', '').replace('[CLS] ', '')
        # print("'" + text[char_start:char_end] + "'")

        found_entity_for_current_mention = False
        for ent_start, ent_len, ent_text in entities:
            ent_end = ent_start + ent_len
            # if ent_start <= char_start < ent_end or ent_start < char_end <= ent_end:
            # if ent_start == char_start and char_end == ent_end:
            if ent_start >= char_start and ent_end <= char_end:
                # if found_entity_for_current_mention:
                #     print('Found more than 1 linked entity for a mention', )
                #     print('LINKED ENTITY: ', ent_text)
                #     print('Mention text: ', text[char_start:char_end])
                #     print(char_start, char_end, ent_start, ent_end)
                found_entity_for_current_mention = True
                entities_found.add(ent_text)

        if found_entity_for_current_mention:
            found_entity_in_cluster = True

    if output_linked_entities:
        return entities_found
    else:
        return found_entity_in_cluster


def print_per_batch(example_ind, is_print, gold_clusters, input_ids, predicted_clusters,
count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, count_missed_mentions,
count_missed_pronouns, count_excess_pronous, count_excess_mentions, tokenizer):
    gold, gold_correct, pred, pred_correct, pred_to_most_similar_gold, pred_to_most_similar_golds_list, gold_is_completely_missed, gold_to_most_similar_pred = match_clusters(
        gold_clusters, predicted_clusters)

    pred_is_completely_missed = [similar_pred == -1 for similar_pred in pred_to_most_similar_gold]
    real_input_ids = [t for t in input_ids.reshape(-1) if t != 1]
    tokens = tokenizer.convert_ids_to_tokens(real_input_ids)
    tokens = [t.replace('Ġ', '') for t in tokens]
    tokens = [t.replace('<pad>', '') for t in tokens]


    #### 1
    count_clusters += len(gold)
    count_mentions += sum(len(c) for c in gold)

    for gold_cluster in gold:
        seen_pronoun = False
        for start,end in gold_cluster:
            mention = tokens[start:end+1]
            if " ".join(mention).lower() in PRONOUNS:
                count_pronouns_mentions += 1
                seen_pronoun = True

        if seen_pronoun:
            count_clusters_with_pronoun_mention += 1
    
    #### 2
    seen_preds = set()
    # calc gold to pred
    for i, gold_cluster in enumerate(gold):
        gold_cluster = set(gold_cluster)
        pred_idx = gold_to_most_similar_pred[i]
        matched_pred = set() if pred_idx is None or pred_idx==-1 else set(pred[pred_idx])
        if pred_idx is not None:
            seen_preds.add(pred_idx)
        diff = gold_cluster - matched_pred
        count_missed_mentions += len(diff)

        diff_text = {" ".join(tokens[start:end + 1]).lower() for start, end in diff}
        count_missed_pronouns += len(diff_text.intersection(PRONOUNS))

        # diff = matched_pred - gold_cluster
        # count_excess_mentions += len(diff)
        # diff_text = {" ".join(tokens[start:end + 1]).lower() for start, end in diff}
        # count_excess_pronous += len(diff_text.intersection(pronouns))


    for i, pred_cluster, in enumerate(pred):
        pred_cluster = set(pred_cluster)
        gold_idx_list = pred_to_most_similar_golds_list[i]

        if len(gold_idx_list) == 0:
            count_excess_mentions += len(pred_cluster)
        else:
            gold_idx = max(gold_idx_list, key=lambda idx: len(set(gold[idx]).intersection(pred_cluster)))

            matched_gold = set(gold[gold_idx])
            diff = pred_cluster - matched_gold
            count_excess_mentions += len(diff)

            diff = {" ".join(tokens[start:end + 1]).lower() for start, end in diff}
            count_excess_pronous += len(diff.intersection(PRONOUNS))

    #### 3
    # entities_per_example = pickle.load(open("/home/gamir/adiz/Code/dev_entities_per_example.p", "rb"))
    # entities_per_sentence = entities_per_example[example['example_num']]
    
    # gold_clusters_to_entities = [is_cluster_contains_linked_entities(c, entities_per_sentence, tokens, True) for c in gold]
    gold_clusters_to_entities = None
    # predicted_clusters_to_entities = [is_cluster_contains_linked_entities(c, entities_per_sentence, tokens, True) for c in pred]
    predicted_clusters_to_entities = None

    # if not (all(gold_correct) and all(pred_correct)):
    # tokens_f = flatten(tokens)
    if is_print:
        print("EXAMPLE #{}".format(example_ind))
        print("GOLD CLUSTERS:")
        gold_cluster_to_color = coref_pprint(tokens, gold, gold_correct, complete_miss_flags=gold_is_completely_missed, cluster_to_linked_entities=gold_clusters_to_entities)

        pred_cluster_to_color = {}
        for pred_i, similar_gold_i in enumerate(pred_to_most_similar_gold):
            if similar_gold_i >= 0 and not gold_correct[similar_gold_i]:
                similar_color = gold_cluster_to_color[gold[similar_gold_i]]
                pred_cluster_to_color[pred[pred_i]] = similar_color

        print("PREDICTED CLUSTERS:")
        coref_pprint(tokens, pred, pred_correct, pred_cluster_to_color, pred_is_completely_missed, cluster_to_linked_entities=predicted_clusters_to_entities)

        # entities_set = set((e for sent_ent in entities_per_sentence for (_,_,e) in sent_ent))
        # print("LINKED ENTITIES SET: ", entities_set)

        print('===========================================================================================================================================================================================================================')
    return count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, \
        count_missed_mentions, count_missed_pronouns, count_excess_pronous, count_excess_mentions




def print_predictions(all_predicted_clusters, all_gold_clusters, all_input_ids, args, tokenizer):

    count_clusters = 0
    count_mentions = 0
    
    count_pronouns_mentions = 0
    count_clusters_with_pronoun_mention = 0
    
    count_missed_mentions = 0
    count_missed_pronouns = 0
    count_excess_mentions = 0
    count_excess_pronous = 0

    indices_to_print = []

    if len(all_input_ids) > args.max_eval_print:
        indices_to_print = np.random.randint(len(all_input_ids), size=args.max_eval_print)
    else:
        indices_to_print = range(len(all_input_ids))

    for i, input_ids in enumerate(all_input_ids):
        count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, \
            count_missed_mentions, count_missed_pronouns, count_excess_pronous, count_excess_mentions = print_per_batch(i, i in indices_to_print,
            all_gold_clusters[i], input_ids, all_predicted_clusters[i],
            count_clusters, count_mentions, count_pronouns_mentions, count_clusters_with_pronoun_mention, count_missed_mentions,
            count_missed_pronouns, count_excess_pronous, count_excess_mentions, tokenizer)

    print("{} gold clusters".format(count_clusters))
    print("{} gold mentions".format(count_mentions))
    print("{} pronouns mentions".format(count_pronouns_mentions))
    print("{} gold clusters with pronouns mentions".format(count_clusters_with_pronoun_mention))


    print("{} missed mentions".format(count_missed_mentions))
    print("{} missed pronouns".format(count_missed_pronouns))
    print("{}% missed pronouns".format(0 if count_missed_mentions == 0 else 1. * count_missed_pronouns / count_missed_mentions * 100))
    print("{} excess mentions".format(count_excess_mentions)) #TODO: wierd numbers
    print("{} excess pronouns".format(count_excess_pronous)) #TODO: wierd numbers
    print("{}% excess pronouns".format(0 if count_excess_mentions == 0 else 1. * count_excess_pronous / count_excess_mentions * 100))


def error_analysis(all_predicted_clusters, all_gold_clusters):
    total_sub_clusters_gold = 0
    total_sub_clusters_pred = 0
    total_num_gold_clusters_in_one_pred_cluster = 0
    total_num_pred_clusters_in_one_gold_cluster = 0
    total_sum_num_split_gold_clusters = 0
    total_sum_num_split_pred_clusters = 0
    total_sum_biggest_prec_gold_cluster_in_pred_cluster = 0
    total_sum_biggest_prec_pred_cluster_in_gold_cluster = 0

    for i in range(len(all_gold_clusters)):
        num_gold_clusters_in_one_pred_cluster, num_pred_clusters_in_one_gold_cluster, \
        sum_num_split_gold_clusters, sum_num_split_pred_clusters, sum_biggest_prec_gold_cluster_in_pred_cluster, \
            sum_biggest_prec_pred_cluster_in_gold_cluster, total_gold_clusters, total_pred_clusters = clusters_error_analysis(
            all_gold_clusters[i], all_predicted_clusters[i][0])
        total_sub_clusters_gold += total_gold_clusters
        total_sub_clusters_pred += total_pred_clusters
        total_num_gold_clusters_in_one_pred_cluster += num_gold_clusters_in_one_pred_cluster
        total_num_pred_clusters_in_one_gold_cluster += num_pred_clusters_in_one_gold_cluster
        total_sum_num_split_gold_clusters += sum_num_split_gold_clusters
        total_sum_num_split_pred_clusters += sum_num_split_pred_clusters
        total_sum_biggest_prec_gold_cluster_in_pred_cluster += sum_biggest_prec_gold_cluster_in_pred_cluster
        total_sum_biggest_prec_pred_cluster_in_gold_cluster += sum_biggest_prec_pred_cluster_in_gold_cluster

    num1 = total_num_gold_clusters_in_one_pred_cluster * 100.0 / total_sub_clusters_gold if \
        total_sub_clusters_gold > 0 else 0
    num2 = total_num_pred_clusters_in_one_gold_cluster * 100.0 / total_sub_clusters_pred if \
        total_sub_clusters_pred > 0 else 0
    num3 = total_sum_num_split_gold_clusters * 1.0 / (total_sub_clusters_gold-total_num_gold_clusters_in_one_pred_cluster) if \
        total_sub_clusters_gold > total_num_gold_clusters_in_one_pred_cluster else 0
    num4 = (total_sum_num_split_gold_clusters+total_num_gold_clusters_in_one_pred_cluster) * 1.0 / total_sub_clusters_gold if \
        total_sub_clusters_gold > 0 else 0
    num5 = total_sum_num_split_pred_clusters * 1.0 / (total_sub_clusters_pred-total_num_pred_clusters_in_one_gold_cluster) if \
        total_sub_clusters_pred > total_num_pred_clusters_in_one_gold_cluster else 0
    num6 = (total_sum_num_split_pred_clusters+total_num_pred_clusters_in_one_gold_cluster) * 1.0 / total_sub_clusters_pred if \
        total_sub_clusters_pred > 0 else 0
    num7 = total_sum_biggest_prec_gold_cluster_in_pred_cluster * 100.0 / (total_sub_clusters_gold-total_num_gold_clusters_in_one_pred_cluster) if \
        total_sub_clusters_gold > total_num_gold_clusters_in_one_pred_cluster else 0
    num8 = (total_sum_biggest_prec_gold_cluster_in_pred_cluster+total_num_gold_clusters_in_one_pred_cluster) * 100.0 / total_sub_clusters_gold if \
        total_sub_clusters_gold > 0 else 0
    num9 = total_sum_biggest_prec_pred_cluster_in_gold_cluster * 100.0 / (total_sub_clusters_pred-total_num_pred_clusters_in_one_gold_cluster) if \
        total_sub_clusters_pred > total_num_pred_clusters_in_one_gold_cluster else 0
    num10 = (total_sum_biggest_prec_pred_cluster_in_gold_cluster+total_num_pred_clusters_in_one_gold_cluster) * 100.0 / total_sub_clusters_pred if \
        total_sub_clusters_pred > 0 else 0
        
    print("{}% gold clusters who went to one pred cluster".format(num1))
    print("{}% pred clusters who containes one gold cluster".format(num2))
    print("{} avg amount of clusters that one gold cluster split to (without clusters who didnt split)".format(num3))
    print("{} avg amount of clusters that one gold cluster split to (with clusters who didnt split)".format(num4))
    print("{} avg amount of gold clusters that one pred cluster containes (without clusters who didnt split)".format(num5))
    print("{} avg amount of gold clusters that one pred cluster containes (with clusters who didnt split)".format(num6))
    print("{}% avg size of biggest sub cluster in gold which is together in pred (without clusters who didnt split)".format(num7)) #TODO: wierd numbers
    print("{}% avg size of biggest sub cluster in gold which is together in pred (with clusters who didnt split)".format(num8)) #TODO: wierd numbers
    print("{}% avg size of biggest sub cluster in pred which is together in gold (without clusters who didnt split)".format(num9)) #TODO: wierd numbers
    print("{}% avg size of biggest sub cluster in pred which is together in gold (with clusters who didnt split)".format(num10)) #TODO: wierd numbers
    return num1, num2, num3, num4, num5, num6, num7, num8, num9, num10


def create_mention_to_cluster_mapping(mentions, clusters):
    target_mention_to_cluster_id = {tuple(m):i for i in range(len(clusters)) for m in clusters[i]}
    missing_mentions = []
    mentions_mapping =[]
    for c in mentions:
        ids = []
        missing_mentions_current = []
        for m in c:
            m = tuple(m)
            if m in target_mention_to_cluster_id:
                ids.append(target_mention_to_cluster_id[m])
            else:
                missing_mentions_current.append(m)
        mentions_mapping.append(ids)
        missing_mentions.append(missing_mentions_current)
    return mentions_mapping, missing_mentions


def clusters_error_analysis(gold, pred):
    gold_mentions_to_pred, missing_mentions_in_pred = create_mention_to_cluster_mapping(gold, pred)
    pred_mentions_to_gold, missing_mentions_in_gold = create_mention_to_cluster_mapping(pred, gold)
    num_gold_clusters_in_one_pred_cluster = 0
    num_pred_clusters_in_one_gold_cluster = 0
    sum_num_split_gold_clusters = 0
    sum_num_split_pred_clusters = 0
    sum_biggest_prec_gold_cluster_in_pred_cluster = 0
    sum_biggest_prec_pred_cluster_in_gold_cluster = 0
    total_gold_clusters = len(gold)
    total_pred_clusters = len(pred)

    for i in range(len(gold)):
        if len(gold_mentions_to_pred[i]) > 0:
            set_g = set(gold_mentions_to_pred[i])
            if len(set_g) == 1 and len(gold[i]) == len(pred[gold_mentions_to_pred[i][0]]) and len(missing_mentions_in_pred[i]) == 0:
                num_gold_clusters_in_one_pred_cluster += 1
            else:
                if len(set_g) == 1:
                    sum_num_split_gold_clusters += len(gold_mentions_to_pred[i]) * 1.0 / len(pred[gold_mentions_to_pred[i][0]])
                else:
                    sum_num_split_gold_clusters += len(set_g)
                c = Counter(gold_mentions_to_pred[i])
                sum_biggest_prec_gold_cluster_in_pred_cluster += c.most_common(1)[0][1] * 1.0 / len(gold_mentions_to_pred[i])
    for j in range(len(pred)):
        if len(pred_mentions_to_gold[j]) > 0:
            set_p = set(pred_mentions_to_gold[j])
            if len(set_p) == 1 and len(pred[j]) == len(gold[pred_mentions_to_gold[j][0]]) and len(missing_mentions_in_gold[j]) == 0:
                num_pred_clusters_in_one_gold_cluster += 1
            else:
                if len(set_p) == 1:
                    sum_num_split_pred_clusters += len(pred_mentions_to_gold[j]) * 1.0 / len(gold[pred_mentions_to_gold[j][0]])
                else:
                    sum_num_split_pred_clusters += len(set_p)
                c = Counter(pred_mentions_to_gold[j])
                sum_biggest_prec_pred_cluster_in_gold_cluster += c.most_common(1)[0][1] * 1.0 / len(pred_mentions_to_gold[j])

    return num_gold_clusters_in_one_pred_cluster, num_pred_clusters_in_one_gold_cluster, \
        sum_num_split_gold_clusters, sum_num_split_pred_clusters, \
            sum_biggest_prec_gold_cluster_in_pred_cluster, sum_biggest_prec_pred_cluster_in_gold_cluster, \
                total_gold_clusters, total_pred_clusters