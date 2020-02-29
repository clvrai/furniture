from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.distributions import FixedCategorical, FixedNormal, \
    Identity, MixedDistribution
from rl.policies.utils import MLP
from util.pytorch import to_tensor


class Actor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__()
        self._config = config
        self._activation_fn = getattr(F, config.rl_activation)
        self._tanh = tanh_policy
        self._gaussian = config.rl_gaussian

    @property
    def info(self):
        return {}

    def act(self, ob, is_train=True, return_log_prob=False):
        ob = to_tensor(ob, self._config.device)
        self._ob = ob
        means, stds = self.forward(ob)

        dists = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                if self._gaussian:
                    dists[k] = FixedNormal(means[k], stds[k])
                else:
                    dists[k] = Identity(means[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        actions = OrderedDict()
        mixed_dist = MixedDistribution(dists)
        if not is_train:
            activations = mixed_dist.mode()
        else:
            activations = mixed_dist.sample()

        if return_log_prob:
            log_probs = mixed_dist.log_probs(activations)

        for k in self._ac_space.keys():
            z = activations[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                if return_log_prob:
                    # follow the Appendix C. Enforcing Action Bounds
                    log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                    log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action.detach().cpu().numpy().squeeze(0)
            activations[k] = z.detach().cpu().numpy().squeeze(0)

        if return_log_prob:
            log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
            if log_probs_.min() < -100:
                print('sampling an action with a probability of 1e-100')
                import ipdb; ipdb.set_trace()

            log_probs_ = log_probs_.detach().cpu().numpy().squeeze(0)
            return actions, activations, log_probs_
        else:
            return actions, activations

    def act_log(self, ob, activations=None):
        self._ob = ob.copy()
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample() if activations is None else activations
        for k in activations_.keys():
            if len(activations_[k].shape) == 1:
                activations_[k] = activations_[k].unsqueeze(0)
        log_probs = mixed_dist.log_probs(activations_)

        for k in self._ac_space.keys():
            z = activations_[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        ents = mixed_dist.entropy()
        log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
        if activations is None:
            return actions, log_probs_
        else:
            return log_probs_, ents

    def act_backprop(self, ob):
        self._ob = ob.copy()
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                if self._gaussian:
                    dists[k] = FixedNormal(means[k], stds[k])
                else:
                    dists[k] = Identity(means[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample()
        for k in activations_.keys():
            if len(activations_[k].shape) == 1:
                activations_[k] = activations_[k].unsqueeze(0)

        for k in self._ac_space.keys():
            z = activations_[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
            else:
                action = z

            actions[k] = action

        return actions

    def act_log_debug(self, ob, activations=None):
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample() if activations is None else activations
        log_probs = mixed_dist.log_probs(activations_)

        for k in self._ac_space.keys():
            z = activations_[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        ents = mixed_dist.entropy()
        log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
        if log_probs_.min() < -100:
            print(ob)
            print(log_probs_.min())
            import ipdb; ipdb.set_trace()
        if activations is None:
            return actions, log_probs_
        else:
            return log_probs_, ents, log_probs, means, stds


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

