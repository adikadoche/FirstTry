SPEAKER_START_ID = 49518  # 'Ġ#####'
SPEAKER_END_ID = 22560  # 'Ġ###'
SPEAKER_START_TOKEN = ' #####'  # 'Ġ#####'
SPEAKER_END_TOKEN = ' ###'  # 'Ġ###'
NULL_ID_FOR_COREF = 0
TOKENS_PAD = 1
SPEAKER_PAD = 0
TOKENS_END = 2
TOKENS_START = 0
MASK_PAD = 0

OUT_KEYS = ['pred_logits', 'pred_clusters', 'pred_is_cluster', 'aux_outputs']

PRONOUNS = {'i', 'me', 'my', 'mine', 'myself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themself', 'themselves',
            'this', 'these', 'that', 'those'}
            
GENRES =  {g: i+1 for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}