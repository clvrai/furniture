from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from il.bc_dataset import ILDataset
from rl.base_agent import BaseAgent
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor


class BCAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, actor, critic):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._actor = actor(self._config, self._ob_space, self._ac_space,
                            self._config.tanh_policy)
        self._network_cuda(config.device)
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_bc)

        self._dataset = ILDataset(config.demo_path)
        self._data_loader = torch.utils.data.DataLoader(self._dataset,
                                                        batch_size=self._config.batch_size,
                                                        shuffle=True)
        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a BC agent')
            logger.info('The actor has %d parameters', count_parameters(self._actor))

    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)

    def sync_networks(self):
        sync_networks(self._actor)

    def train(self):
        train_info = {}
        for transitions in self._data_loader:
            _train_info = self._update_network(transitions)
            train_info.update(_train_info)

        train_info.update({
            'actor_grad_norm': np.mean(compute_gradient_norm(self._actor)),
            'actor_weight_norm': np.mean(compute_weight_norm(self._actor)),
        })
        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o = transitions['ob']
        o = self.normalize(o)

        bs = len(transitions['ac'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions['ac'])

        # the actor loss
        pred_ac = self._actor.act_backprop(o)
        if isinstance(pred_ac, OrderedDict):
            pred_ac = list(pred_ac.values())
            if len(pred_ac[0].shape) == 1:
                pred_ac = [x.unsqueeze(0) for x in pred_ac]
            pred_ac = torch.cat(pred_ac, dim=-1)

        actor_loss = (ac - pred_ac).pow(2).mean()
        info['actor_loss'] = actor_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        return mpi_average(info)


