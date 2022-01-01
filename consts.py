import string

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

LETTERS = string.ascii_lowercase
LETTERS_LIST = list(LETTERS)

BACKBONE_PATHS = {
    'ontonotes': {
        'gold': {
            'junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_05_emb_junk_gold_ontonotes/coref-detr/fg8u0bkr/checkpoints/epoch=39-step=110999.ckpt', 
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_05_emb_junk_gold_ontonotes/coref-detr/fg8u0bkr/checkpoints/epoch=7-step=22199.ckpt' #
            },
            'no_junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_05_emb_gold_ontonotes/coref-detr/eqh43bnx/checkpoints/epoch=39-step=110999.ckpt',
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_05_emb_gold_ontonotes/coref-detr/eqh43bnx/checkpoints/epoch=3-step=11099.ckpt'#
            }
        },
        'no_gold': {
            'no_junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_36_emb_ontonotes/coref-detr/t9xwi1o2/checkpoints/epoch=39-step=110999.ckpt',
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_00_36_emb_ontonotes/coref-detr/t9xwi1o2/checkpoints/epoch=10-step=30524.ckpt'#
            }
        }
    },
    'sequences': {
        'gold': {
            'junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_14_32_emb_junk_gold_seq9950/coref-detr/1rp3yhyb/checkpoints/epoch=D14-step=134324.ckpt', #
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_14_32_emb_junk_gold_seq9950/coref-detr/1rp3yhyb/checkpoints/epoch=D14-step=134324.ckpt' #
            },
            'no_junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_01_15_27_emb_gold_seq9950/coref-detr/236k5wd2/checkpoints/epoch=7-step=71639.ckpt',
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_01_15_27_emb_gold_seq9950/coref-detr/236k5wd2/checkpoints/epoch=4-step=44774.ckpt'
            }
        },
        'no_gold': {
            'no_junk': {
                'latest': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_15_00_emb_seq9950/coref-detr/dj7mlm9p/checkpoints/epoch=14-step=134324.ckpt',
                'best': '/home/gamir/adiz/Code/runs/firsttry/output_dir/12_31_2021_09_15_00_emb_seq9950/coref-detr/dj7mlm9p/checkpoints/epoch=14-step=134324.ckpt' 
            }
        }
    }
}