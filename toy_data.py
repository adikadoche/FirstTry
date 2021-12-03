import random
import string

def create_letters_dataser(num_of_texts = 3000):
    text = ''
    clusters = []
    for _ in range(num_of_texts):
        text_len = random.randint(40, 4000)
        line_list = random.choices(string.ascii_lowercase, k=text_len) #change to non equal distribution

        text += ' '.join(line_list) +'\n'

        letters_indices = {}
        text_cluster = []
        start_index = 0
        last_letter = line_list[0]
        for i, letter in enumerate(line_list):
            if letter not in letters_indices.keys():
                letters_indices[letter] = len(letters_indices)
                text_cluster.append([])
            if letter == last_letter:
                continue
            if letter != last_letter:
                text_cluster[letters_indices[last_letter]].append([start_index, i-1])
                last_letter = letter
                start_index = i
        text_cluster[letters_indices[last_letter]].append([start_index, len(line_list)-1])
        
        clusters.append(text_cluster)

    return text, clusters

create_letters_dataser()