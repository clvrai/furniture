import numpy as np
from gym import spaces


def observation_size(observation_space):
    if isinstance(observation_space, spaces.Dict):
        return sum([observation_size(value) for key, value in observation_space.spaces.items()])
    elif isinstance(observation_space, spaces.Box):
        return np.product(observation_space.shape)


def action_size(action_space):
    if isinstance(action_space, spaces.Dict):
        return sum([action_size(value) for key, value in action_space.spaces.items()])
    elif isinstance(action_space, spaces.Box):
        return np.product(action_space.shape)
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return np.product(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n


