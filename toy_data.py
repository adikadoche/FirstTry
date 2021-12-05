from collections import namedtuple
import random
import string

LETTERS = string.ascii_lowercase
LETTERS_LIST = list(LETTERS)

def create_letters_dataset(num_of_texts = 3000):
    text = ''
    clusters = []
    for _ in range(num_of_texts):
        text_len = random.randint(40, 4000)
        bkgd_text_len = int(random.uniform(0.5, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.randint(1, len(LETTERS)-5)
        cur_letters = random.sample(LETTERS_LIST, num_clusters)
        bgkd_letters = [l for l in LETTERS if l not in cur_letters]
        sequence_distribution = [random.uniform(0, 1) for _ in range(num_clusters)]
        line_sequence_list = random.choices(cur_letters, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(bgkd_letters))]
        line_bkgd_list = random.choices(bgkd_letters, weights=bgkd_distribution, k=bkgd_text_len)

        sorted_pairs = [(x, y) for x, y in sorted(zip(line_sequence_list, sequence_indices), key=lambda pair: pair[1])]

        line_list = line_bkgd_list

        for letter, index in reversed(sorted_pairs):
            line_list.insert(index, letter)        

        text += ' '.join(line_list) +'\n'

        letters_indices = {}
        text_cluster = []
        for i, letter in enumerate(line_list):
            if letter in cur_letters:
                if letter not in letters_indices.keys():
                    letters_indices[letter] = len(letters_indices)
                    text_cluster.append([])
                text_cluster[letters_indices[letter]].append([i, i])
        
        clusters.append(text_cluster)

    return text, clusters

def create_sequences_dataset(num_of_texts = 3000):
    SEQUENCES = []
    for _ in range(70):
        seq_len = random.randint(1, 7)
        seq = random.choices(LETTERS_LIST, k=seq_len)
        if seq not in SEQUENCES:
            SEQUENCES.append(seq)
    text = ''
    clusters = []
    for t in range(num_of_texts):
        clusters.append([])
        text_len = random.randint(40, 4000)
        bkgd_text_len = int(random.uniform(0.5, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.randint(1, min(int(text_len/4), len(SEQUENCES)))
        cur_sequenceds = random.sample(SEQUENCES, num_clusters)
        sequence_distribution = [random.uniform(0, 1) for _ in range(len(cur_sequenceds))]
        sequence_text_len = int(sequence_text_len / (sum([len(cur_sequenceds[i]) * sequence_distribution[i] for i in range(len(cur_sequenceds))]) / sum(sequence_distribution)))
        line_sequence_list = random.choices(cur_sequenceds, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(LETTERS))]
        line_bkgd_list = random.choices(LETTERS, weights=bgkd_distribution, k=bkgd_text_len)

        sorted_pairs = [(x, y) for x, y in sorted(zip(line_sequence_list, sequence_indices), key=lambda pair: pair[1])]

        line_list = line_bkgd_list

        for sequence, index in reversed(sorted_pairs):
            line_list[index:index] = sequence

        for s in cur_sequenceds:
            sequence_cluster = []
            for j in range(len(line_list) - len(s) + 1):
                if line_list[j:j+len(s)] == s:
                    sequence_cluster.append([j, j+len(s)-1])
            if len(sequence_cluster) > 0:
                clusters[t].append(sequence_cluster)

        text += ' '.join(line_list) +'\n'

    return text, clusters

def create_structural_dataset(num_of_texts = 3000):
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
                                                                sequence=random.choices(LETTERS, k=random.randint(1, 3)))
                total_len += sequence_pattern_list[i].index_delta + len(sequence_pattern_list[i].sequence)
        cur_seq_pattern = SequencePattern(left=sequence_pattern_list[0], right=sequence_pattern_list[1], total_len=total_len+1)
        if cur_seq_pattern not in SEQUENCES:
            SEQUENCES.append(cur_seq_pattern)

    text = ''
    clusters = []
    for t in range(num_of_texts):
        clusters.append([])
        text_len = random.randint(40, 4000)
        bkgd_text_len = int(random.uniform(0.5, 1) * text_len)
        sequence_text_len = text_len - bkgd_text_len
        num_clusters = random.randint(1, min(int(text_len/4), len(SEQUENCES)))
        cur_sequenceds = random.sample(SEQUENCES, num_clusters)
        sequence_distribution = [random.uniform(0, 1) for _ in range(len(cur_sequenceds))]
        sequence_text_len = int(sequence_text_len / (sum([cur_sequenceds[i].total_len * sequence_distribution[i] for i in range(len(cur_sequenceds))]) / sum(sequence_distribution)))
        line_sequence_list = random.choices(cur_sequenceds, weights=sequence_distribution, k=sequence_text_len)
        sequence_indices = random.sample(list(range(bkgd_text_len)), sequence_text_len)
        bgkd_distribution = [random.uniform(0, 1) for _ in range(len(LETTERS))]
        line_bkgd_list = random.choices(LETTERS, weights=bgkd_distribution, k=bkgd_text_len)

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

        text += ' '.join(line_list) + '\n'
        for s in cur_sequenceds:
            s_cluster = []
            for i in range(len(line_list) - s.total_len + 1):
                if line_list[i:i+len(s.left.sequence)] == s.left.sequence and \
                    line_list[i+len(s.left.sequence)+s.left.index_delta+1+s.right.index_delta:i+s.total_len] == s.right.sequence:
                    s_cluster.append([i+len(s.left.sequence)+s.left.index_delta, i+len(s.left.sequence)+s.left.index_delta])
            if len(s_cluster) > 0:
                clusters[-1].append(s_cluster)
    return text, clusters
        

create_letters_dataset(20)
create_structural_dataset(20)
create_sequences_dataset(20)