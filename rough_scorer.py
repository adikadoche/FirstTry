""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import List, Tuple

import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers.models.bert.modeling_bert import ACT2FN
import torch.nn.functional as F


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob=0.3):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, config, topk_lambda: float):
        super().__init__()
        self.ffnn_size = 3072
        self.mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size)
        self.mention_classifier = Linear(self.ffnn_size, 1)
        self.k = topk_lambda

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                word_clusters: List
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        mention_reps = self.mention_mlp(mentions)
        mention_logits = self.mention_classifier(mention_reps).squeeze(-1)
        mention_logits = mention_logits.sigmoid()
        top_scores, indices = torch.topk(mention_logits,
                                         k=int(self.k * len(mention_logits)),
                                         dim=0, sorted=False)
        ind_sort = torch.argsort(indices)
        indices = indices[ind_sort]
        top_scores = top_scores[ind_sort]
        gold_indices = [gw for gc in word_clusters for gw in gc]
        cost_is_mention = torch.tensor(0., device=mention_logits.device)
        if len(gold_indices) > 0:
            gold_probs = mention_logits[gold_indices]
            cost_is_mention = F.binary_cross_entropy(gold_probs, torch.ones_like(gold_probs))
        junk_probs = mention_logits[[i for i in range(len(mention_logits)) if i not in gold_indices]]
        cost_is_mention += F.binary_cross_entropy(junk_probs, torch.zeros_like(junk_probs))
        return top_scores, indices, cost_is_mention
