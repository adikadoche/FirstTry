import unittest
import torch
import random
import numpy as np
from matcher import HungarianMatcher
from utils import calc_predicted_clusters
from detr import DETRLoss


num_queries = 50
overlap_clusters = [[[(62, 62), (86, 86), (89, 89), (174, 175), (394, 394), (532, 533), (547, 547)],\
    [(283, 284), (335, 340)],\
    [(184, 187), (222, 225)],\
    [(57, 63) , (102, 102), (138, 140), (150, 152), (247, 252), (279, 281), (289, 289), (306, 306), (317, 318), (320, 320), (327, 327), (339, 340), (346, 346), (350, 350), (376,376), (381,383),(415,415),(541,544)],\
    [(159, 159), (164, 165)],\
    [(176, 200), (207, 208), (219, 219), (239, 240)],\
    [(363, 363), (418, 418), (426, 426), (431, 431)],\
    [(34, 34), (38, 43)],\
    [(365, 366), (400, 401)],\
    [(29, 30), (47, 48), (124, 125), (138, 139), (150, 151), (168, 169), (215, 216), (279, 280), (535, 536)],\
    [(227, 230), (234, 234)],\
    [(405, 407), (444, 446)]]]
clusters = [[[(86, 86), (89, 89), (174, 175), (394, 394), (532, 533), (547, 547)],\
    [(283, 284), (335, 340)],\
    [(222, 225)],\
    [(57, 63) , (102, 102), (138, 140), (247, 252), (289, 289), (306, 306), (317, 318), (320, 320), (327, 327), (346, 346), (350, 350), (376,376), (381,383),(415,415),(541,544)],\
    [(159, 159), (164, 165)],\
    [(176, 200), (207, 208), (219, 219), (239, 240)],\
    [(363, 363), (418, 418), (426, 426), (431, 431)],\
    [(34, 34), (38, 43)],\
    [(365, 366), (400, 401)],\
    [(29, 30), (47, 48), (124, 125), (150, 151), (168, 169), (215, 216), (279, 280), (535, 536)],\
    [(227, 230), (234, 234)],\
    [(405, 407), (444, 446)]]]
gold_mentions_list = [[i for j in clusters[0] for i in j]]

clusters_inds = list(range(len(clusters[0])))
random.shuffle(clusters_inds)
cluster_logits = torch.zeros([1, num_queries, 1])
cluster_logits[0][clusters_inds]=1

coref_logits_gold = torch.zeros([1, num_queries, len(gold_mentions_list[0])])
for i, cluster in enumerate(clusters[0]):
    cluster_id = clusters_inds[i]
    for m in cluster:
        coref_logits_gold[0][cluster_id][gold_mentions_list[0].index(tuple(m))] = 1

coref_logits_tokens = torch.zeros([1, num_queries, 561])
for i, cluster in enumerate(clusters[0]):
    cluster_id = clusters_inds[i]
    for m in cluster:
        coref_logits_tokens[0][cluster_id][m[0]:m[1]+1] = 1
coref_target_tokens = coref_logits_tokens.clone()
non_cluster_ind = random.sample([i for i in list(range(num_queries)) if i not in clusters_inds], 1)[0]
token_inds = [list(range(m[0],m[1]+1)) for m in gold_mentions_list[0]]
token_inds = [i for j in token_inds for i in j]
non_mention_token_inds = [i for i in range(561) if i not in token_inds]
for i in non_mention_token_inds:
    coref_logits_tokens[0][non_cluster_ind][i] = 1

class Args:
    def __init__(self, is_cluster, cluster_block, reduction='sum', use_gold_mentions=True, slots=True):
        self.cost_is_cluster=3
        self.cost_coref=1
        self.cost_is_mention=1
        self.add_junk=False
        self.BIO=1
        self.is_cluster = is_cluster
        self.use_gold_mentions = use_gold_mentions
        self.slots = slots
        self.cluster_block = cluster_block
        self.eos_coef=0.1
        self.reduction=reduction
        self.b3_loss=False

class Test(unittest.TestCase):
    def test_calc_predicted(self):
        #calc_predicted_clusters(cluster_logits, coref_logits, mention_logits, coref_threshold, cluster_threshold, mentions: List, use_gold_mentions, use_topk_mentions, is_cluster, slots, min_cluster_size)
        options=[True,False]
        for o1 in options:  #use_gold_mentions
            for o2 in options:   #slots
                if o1:
                    res = calc_predicted_clusters(cluster_logits, coref_logits_gold, [], 0.5, 0.5, gold_mentions_list, o1, False, True, o2, 0)
                else:
                    res = calc_predicted_clusters(cluster_logits, coref_logits_tokens, [], 0.5, 0.5, gold_mentions_list, o1, False, True, o2, 0)
                self.assertCountEqual(set([tuple(c) for c in res[0]]),set([tuple(c) for c in clusters[0]]))

    def test_matcher(self):
        #targets['clusters'],targets['mentions']
        #outputs["coref_logits"],outputs["cluster_logits"],outputs["mention_logits"]
        targets = {}
        targets['mentions'] = []
        outputs = {}
        outputs['cluster_logits'] = cluster_logits
        outputs['mention_logits'] = []
        
        outputs['coref_logits'] = coref_logits_tokens
        targets['clusters'] = coref_logits_tokens
        options=[True,False]
        for i in range(2):  #tokens/gold
            if i==0:
                outputs['coref_logits'] = coref_logits_tokens
                targets['clusters'] = coref_target_tokens
            else:
                outputs['coref_logits'] = coref_logits_gold
                targets['clusters'] = coref_logits_gold
            for o1 in options:  #is cluster
                for o2 in options:   #cluster block
                    args = Args(o1, o2, use_gold_mentions=i==1)
                    m = HungarianMatcher(args=args)
                    p_r,g_r,p_f,g_f = m(outputs,targets)
                    self.assertCountEqual(p_r[0][g_r[0]].numpy(),clusters_inds)

    def test_loss(self):
        #targets['clusters'],targets['mentions']
        #outputs["coref_logits"],outputs["cluster_logits"],outputs["mention_logits"]
        targets = {}
        targets['mentions'] = []
        outputs = {}
        outputs['cluster_logits'] = cluster_logits
        outputs['mention_logits'] = []
        
        options=[True,False]
        reduc_options=['sum','mean']
        for i in range(2):  #tokens/gold
            if i==0:
                outputs['coref_logits'] = coref_logits_tokens
                targets['clusters'] = coref_target_tokens
            else:
                outputs['coref_logits'] = coref_logits_gold
                targets['clusters'] = coref_logits_gold
            for o1 in options:  #is cluster
                for o2 in options:   #cluster block
                    for o3 in options:   #slots/DETR
                        for r in reduc_options:   #reduction]
                            if not o3 and i==0:
                                continue
                            args = Args(o1, o2, r, use_gold_mentions=i==1, slots=o3)
                            m = HungarianMatcher(args=args)
                            l = DETRLoss(m, args.eos_coef, args.cost_is_cluster, args.cost_coref, args.cost_is_mention, args)
                            loss,parts = l(outputs,targets)
                            if loss != 0:
                                print(i,o1,o2,r)
                            self.assertEqual(loss,0)

if __name__ == '__main__':
    unittest.main()