import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from il.base_agent import BaseAgent
from util.logger import logger
from util.mpi import mpi_average
from il.BCDataset import BCDataset
from il.BCModel import *
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor


class BCAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space):
        super().__init__(config, ob_space)
        self._config = config

        self.net = getBCModel(ob_space, ac_space, weights_path=self._config.saved_weights)
        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            # logger.info('Creating a DDPG agent')
            # logger.info('The actor has %d parameters', count_parameters(self._actor))
            # logger.info('The critic has %d parameters', count_parameters(self._critic))

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def act(self, ob, is_train=False):
        actions = self._net(ob)
        return actions

        