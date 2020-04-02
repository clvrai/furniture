import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = getattr(F, config.activation)

        self.convs = nn.ModuleList()
        d_prev = 3
        d = config.conv_dim
        w = config.screen_width
        for k, s in zip(config.kernel_size, config.stride):
            self.convs.append(nn.Conv2d(d_prev, d, int(k), int(s)))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            d_prev = d

        # screen_width == 32 (8,4)-(3,2) -> 3x3
        # screen_width == 64 (8,4)-(3,2)-(3,2) -> 3x3
        # screen_width == 128 (8,4)-(3,2)-(3,2)-(3,2) -> 3x3
        # screen_width == 256 (8,4)-(3,2)-(3,2)-(3,2) -> 7x7

        print('Output of CNN = %d x %d x %d' % (w, w, d))
        self.output_size = w * w * d

    def forward(self, ob):
        out = ob
        for conv in self.convs:
            out = self.activation_fn(conv(out))
        out = out.flatten(start_dim=1)
        return out


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim, hid_dims=[]):
        super().__init__()
        #activation_fn = getattr(F, config.activation)
        activation_fn = nn.ReLU()

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fanin_init(fc[-1].weight)
            fc[-1].bias.data.fill_(0.1)
            fc.append(activation_fn)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        self.fc = nn.Sequential(*fc)

    def forward(self, ob):
        return self.fc(ob)

