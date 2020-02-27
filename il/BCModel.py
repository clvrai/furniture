import torch
import torch.nn as nn



class BCModel(nn.Module):

    def __init__(self, obs_space, action_space):
        super(BCModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_space)
        )


    def forward(self, x):
        x = self.features(x)
        return x


def getBCModel(obs_space, action_space, weights_path=None, progress=True, **kwargs):
    model = BCModel(obs_space, action_space, **kwargs).double()
    if weights_path:
        model.load_state_dict(weights_path)
    return model