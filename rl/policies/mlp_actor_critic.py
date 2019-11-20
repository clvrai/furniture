from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from rl.policies.utils import CNN, MLP
from rl.policies.actor_critic import Actor, Critic


class MlpActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        self._ac_space = ac_space

        # observation
        input_dim = sum([np.prod(x) for x in ob_space.values()])

        self.fc = MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size])
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, size in ac_space.shape.items():
            self.fc_means.update({k: MLP(config, config.rl_hid_size, size)})
            if ac_space.is_continuous(k):
                self.fc_log_stds.update({k: MLP(config, config.rl_hid_size, size)})

    def forward(self, ob):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]

        out = self._activation_fn(self.fc(torch.cat(inp, dim=-1)))
        out = torch.reshape(out, (out.shape[0], -1))

        means, stds = OrderedDict(), OrderedDict()
        for k in self._ac_space.keys():
            mean = self.fc_means[k](out)
            if self._ac_space.is_continuous(k):
                log_std = self.fc_log_stds[k](out)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None

            means[k] = mean
            stds[k] = std

        return means, stds


class MlpCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        input_dim = sum([np.prod(x) for x in ob_space.values()])
        if ac_space is not None:
            input_dim += ac_space.size

        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2)

    def forward(self, ob, ac=None):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]

        if ac is not None:
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            inp.extend(ac)

        out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out

