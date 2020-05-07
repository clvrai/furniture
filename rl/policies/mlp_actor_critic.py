from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import gym.spaces

from rl.policies.utils import MLP, flatten_ob
from rl.policies.actor_critic import Actor, Critic
from util.pytorch import to_tensor
from util.gym import observation_size, action_size


class MlpActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        # observation
        input_dim = observation_size(ob_space)

        self.fc = MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size])
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, v in ac_space.spaces.items():
            self.fc_means.update({k: MLP(config, config.rl_hid_size, action_size(v))})
            if isinstance(v, gym.spaces.Box) and self._deterministic:
                self.fc_log_stds.update(
                    {k: MLP(config, config.rl_hid_size, action_size(v))}
                )

    def forward(self, ob: dict):
        # flatten image before concatenating with other obs
        # first check if we are doing a single instance or batch
        inp = flatten_ob(ob)

        out = self._activation_fn(self.fc(inp))
        out = torch.reshape(out, (out.shape[0], -1))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            mean = self.fc_means[k](out)
            if isinstance(v, gym.spaces.Box) and self._deterministic:
                log_std = self.fc_log_stds[k](out)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None

            means[k] = mean
            stds[k] = std

        return means, stds


class NoisyMlpActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        # observation
        input_dim = observation_size(ob_space)

        self.fc = MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size])
        self.fc_means = nn.ModuleDict()

        for k, v in ac_space.spaces.items():
            self.fc_means.update({k: MLP(config, config.rl_hid_size, action_size(v))})

    def forward(self, ob):
        inp = flatten_ob(ob)

        out = self._activation_fn(self.fc(inp))
        out = torch.reshape(out, (out.shape[0], -1))

        means = OrderedDict()
        for k in self._ac_space.spaces.keys():
            means[k] = self.fc_means[k](out)

        return means, None

    def act(self, ob, is_train=True, return_log_prob=False):
        ob = to_tensor(ob, self._config.device)
        self._ob = ob
        means, _ = self.forward(ob)

        actions = OrderedDict()
        for k in self._ac_space.spaces.keys():
            z = means[k]
            if self._tanh:
                action = torch.tanh(z)
            else:
                action = z

            ac = action.detach().cpu().numpy().squeeze(0)
            act = z.detach().cpu().numpy().squeeze(0)
            if is_train:
                ac += self._config.noise_eps * np.random.randn(*ac.shape)
                ac = np.clip(ac, -1, 1)
                if np.random.uniform() < self._config.random_eps:
                    ac = np.random.uniform(low=-1, high=1, size=len(ac))

            actions[k] = ac

        if return_log_prob:
            raise NotImplementedError()
        else:
            return actions, None


class MlpCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        # observation
        input_dim = sum([np.prod(x) for x in ob_space.values()])
        if ac_space is not None:
            input_dim += action_size(ac_space)

        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2)

    def forward(self, ob, ac=None):
        inp = flatten_ob(ob, ac)
        out = self.fc(inp)
        out = torch.reshape(out, (out.shape[0], 1))

        return out
