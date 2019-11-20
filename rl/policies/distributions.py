from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.distributions


# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
#FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

categorical_entropy = FixedCategorical.entropy
FixedCategorical.entropy = lambda self: categorical_entropy(self) * 10.0 # scaling

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)


# Normal
FixedNormal = torch.distributions.Normal

normal_init = FixedNormal.__init__
FixedNormal.__init__ = lambda self, mean, std: normal_init(self, mean.double(), std.double())

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions.double()).sum(-1, keepdim=True).float()

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1).float()

FixedNormal.mode = lambda self: self.mean.float()

normal_sample = FixedNormal.sample
FixedNormal.sample = lambda self: normal_sample(self).float()

normal_rsample = FixedNormal.rsample
FixedNormal.rsample = lambda self: normal_rsample(self).float()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logstd = AddBias(torch.zeros(config.action_size))
        self.config = config

    def forward(self, x):
        zeros = torch.zeros(x.size()).to(self.config.device)
        logstd = self.logstd(zeros)
        return FixedNormal(x, logstd.exp())


class MixedDistribution(nn.Module):
    def __init__(self, distributions):
        super().__init__()
        assert isinstance(distributions, OrderedDict)
        self.distributions = distributions

    def mode(self):
        return OrderedDict(
            [(k, dist.mode()) for k, dist in self.distributions.items()]
        )

    def sample(self):
        return OrderedDict(
            [(k, dist.sample()) for k, dist in self.distributions.items()]
        )

    def rsample(self):
        return OrderedDict(
            [(k, dist.rsample()) for k, dist in self.distributions.items()]
        )

    def log_probs(self, x):
        assert isinstance(x, dict)
        return OrderedDict(
            [(k, dist.log_probs(x[k])) for k, dist in self.distributions.items()]
        )

    def entropy(self):
        return sum([dist.entropy() for dist in self.distributions.values()])

