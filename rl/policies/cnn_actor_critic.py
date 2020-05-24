from collections import OrderedDict

import gym.spaces
import numpy as np
import torch
import torch.nn as nn

from rl.policies.actor_critic import Actor, Critic
from rl.policies.utils import CNN, MLP, flatten_ac


class Encoder(nn.Module):
    def __init__(self, config, ob_space):
        super().__init__()

        self._encoder_type = config.encoder_type
        self._ob_space = ob_space

        self.base = nn.ModuleDict()
        encoder_output_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) in [3, 4]:
                if self._encoder_type == "mlp":
                    self.base[k] = None
                    encoder_output_dim += gym.spaces.flatdim(v)
                else:
                    if len(v.shape) == 3:
                        image_dim = v.shape[2]
                    elif len(v.shape) == 4:
                        image_dim = v.shape[0] * v.shape[3]
                    self.base[k] = CNN(config, image_dim)
                    encoder_output_dim += self.base[k].output_dim
            elif len(v.shape) == 1:
                self.base[k] = None
                encoder_output_dim += gym.spaces.flatdim(v)
            else:
                raise ValueError("Check the shape of observation %s (%s)" % (k, v))

        self.output_dim = encoder_output_dim

    def forward(self, ob: dict, detach_cnn=False):
        encoder_outputs = []
        for k, v in ob.items():
            if len(v.shape) == len(self._ob_space.spaces[k].shape):
                if len(v.shape) != 4:
                    v = v.unsqueeze(0)
            if self.base[k] is not None:
                encoder_outputs.append(self.base[k](v, detach_cnn))
            else:
                encoder_outputs.append(v.flatten(start_dim=1))
        out = torch.cat(encoder_outputs, dim=-1)
        assert len(out.shape) == 2
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for k, v in ob.items():
            if self.base[k] is not None:
                self.base[k].copy_conv_weights_from(source.base[k])


class CnnActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        self.encoder = Encoder(config, ob_space)

        self.fc = MLP(
            config, self.encoder.output_dim, config.rl_hid_size, [config.rl_hid_size]
        )
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, v in ac_space.spaces.items():
            self.fc_means.update(
                {k: MLP(config, config.rl_hid_size, gym.spaces.flatdim(v))}
            )
            if isinstance(v, gym.spaces.Box) and not self._deterministic:
                self.fc_log_stds.update(
                    {k: MLP(config, config.rl_hid_size, gym.spaces.flatdim(v))}
                )

    def forward(self, ob: dict):
        # encoder
        out = self.encoder(ob, detach_cnn=True)

        # fc
        out = self._activation_fn(self.fc(out))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            mean = self.fc_means[k](out)
            if isinstance(v, gym.spaces.Box) and not self._deterministic:
                log_std = self.fc_log_stds[k](out)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None

            means[k] = mean
            stds[k] = std

        return means, stds


class CnnCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)
        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2)

    def forward(self, ob, ac=None):
        # encoder
        out = self.encoder(ob, detach_cnn=True)

        if ac is not None:
            out = torch.cat([out, flatten_ac(ac)], dim=-1)
        assert len(out.shape) == 2

        out = self.fc(out)
        out = torch.reshape(out, (out.shape[0], 1))

        return out
