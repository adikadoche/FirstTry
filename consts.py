SPEAKER_START = 49518  # 'Ġ#####'
SPEAKER_END = 22560  # 'Ġ###'
NULL_ID_FOR_COREF = 0

OUT_KEYS = ['pred_logits', 'pred_clusters', 'pred_is_cluster', 'aux_outputs']

PRONOUNS = {'i', 'me', 'my', 'mine', 'myself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themself', 'themselves',
            'this', 'these', 'that', 'those'}