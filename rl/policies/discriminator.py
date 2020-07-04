from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.utils import MLP
from util.pytorch import to_tensor


class Discriminator(nn.Module):
    def __init__(self, config, ob_space, ac_space):
        super().__init__()
        self._config = config

        # observation
        input_dim = sum([np.prod(x) for x in ob_space.values()]) + sum(
            [np.prod(x) for x in ac_space.values()]
        )

        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2)

    def forward(self, ob, ac):
        # flatten observation
        ob = list(ob.values())
        if len(ob[0].shape) == 1:
            ob = [x.unsqueeze(0) for x in ob]
        ob = torch.cat(ob, dim=-1)

        # flatten action
        if isinstance(ac, OrderedDict):
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            ac = torch.cat(ac, dim=-1)

        out = self.fc(torch.cat([ob, ac], dim=-1))
        return out

    def predict_reward(self, ob, ac):
        ob = to_tensor(ob, self._config.device)
        ac = to_tensor(ac, self._config.device)
        ret = self.forward(ob, ac)

        reward = -torch.log(1 - torch.sigmoid(ret) + 1e-8)
        return reward
