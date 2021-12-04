import random
import string

LETTERS = string.ascii_lowercase
LETTERS_LIST = list(LETTERS)

def create_letters_dataser(num_of_texts = 3000):
    text = ''
    clusters = []
    for _ in range(num_of_texts):
        text_len = random.randint(40, 4000)
        num_clusters = random.randint(1, len(LETTERS))
        cur_letters = random.sample(LETTERS_LIST, num_clusters)
        distribution = [random.uniform(0, 1) for _ in range(num_clusters)]
        line_list = random.choices(cur_letters, weights=distribution, k=text_len)

        text += ' '.join(line_list) +'\n'

        letters_indices = {}
        text_cluster = []
        for i, letter in enumerate(line_list):
            if letter not in letters_indices.keys():
                letters_indices[letter] = len(letters_indices)
                text_cluster.append([])
            text_cluster[letters_indices[letter]].append([i, i])
        
        clusters.append(text_cluster)

    return text, clusters

def create_sequences_dataser(num_of_texts = 3000):
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
        num_clusters = random.randint(1, min(int(text_len/4), len(SEQUENCES)))
        cur_sequenceds = random.sample(SEQUENCES, num_clusters)
        text_len = int(text_len / (sum([len(c) for c in cur_sequenceds]) / num_clusters))
        sequence_distribution = [random.uniform(0, 1) for _ in range(len(cur_sequenceds))]
        line_list = random.choices(cur_sequenceds, weights=sequence_distribution, k=text_len)
        line_list = [letter for seq in line_list for letter in seq]

        for s in cur_sequenceds:
            sequence_cluster = []
            for j in range(len(line_list) - len(s) + 1):
                if line_list[j:j+len(s)] == s:
                    sequence_cluster.append([j, j+len(s)-1])
            if len(sequence_cluster) > 0:
                clusters[t].append(sequence_cluster)
            

        text += ' '.join(line_list) +'\n'

    return text, clusters

create_sequences_dataser(20)