from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from il.bc_dataset import ILDataset
from rl.base_agent import BaseAgent
from util.logger import logger
from util.mpi import mpi_average
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiplicativeLR
from util.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
)


class BCAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, actor, critic):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._epoch = 0

        self._actor = actor(
            self._config, self._ob_space, self._ac_space, self._config.tanh_policy
        )
        self._network_cuda(config.device)
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_bc, )
        self._sched = False
        if config.sched_lambda:
            self._sched = True
            schedule = lambda epoch: config.sched_lambda
            self._scheduler = MultiplicativeLR(self._actor_optim, lr_lambda=schedule)

        self._dataset = ILDataset(config.demo_path)

        if self._config.val_split != 0:
            dataset_size = len(self._dataset)
            indices = list(range(dataset_size))
            split = int(np.floor((1-self._config.val_split) * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            self._train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self._config.batch_size, sampler=train_sampler)
            self._val_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self._config.batch_size, sampler=val_sampler)
        else:
            self._train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self._config.batch_size, shuffle=True)

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a BC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))

    def state_dict(self):
        return {
            "actor_state_dict": self._actor.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)

    def sync_networks(self):
        sync_networks(self._actor)

    def train(self):
        train_info = {}
        for transitions in self._train_loader:
            _train_info = self._update_network(transitions, train=True)
            train_info.update(_train_info)
        self._epoch += 1
        if self._sched:
            self._scheduler.step()

        train_info.update(
            {
                "actor_grad_norm": np.mean(compute_gradient_norm(self._actor)),
                "actor_weight_norm": np.mean(compute_weight_norm(self._actor)),
            }
        )
        return train_info

    def evaluate(self):
        if self._val_loader:
            eval_info = {}
            for transitions in self._val_loader:
                _val_info = self._update_network(transitions, train=False)
                val_info.update(_val_info)
            self._epoch += 1
            if self._sched:
                self._scheduler.step()
            return val_info
        logger.warning("No validation set available, make sure '--val_split' is set")
        return None

    def _update_network(self, transitions, train=True):
        info = {}

        # pre-process observations
        o = transitions["ob"]
        o = self.normalize(o)

        # convert double tensor to float32 tensor
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions["ac"])

        # the actor loss
        pred_ac = self._actor.act_backprop(o)
        # print('pred_ac', pred_ac)
        # print('GT ac', ac)
        if isinstance(pred_ac, OrderedDict):
            pred_ac = list(pred_ac.values())
            if len(pred_ac[0].shape) == 1:
                pred_ac = [x.unsqueeze(0) for x in pred_ac]
            pred_ac = torch.cat(pred_ac, dim=-1)

        diff = (ac - pred_ac)
        actor_loss = diff.pow(2).mean()
        info["actor_loss"] = actor_loss.cpu().item()
        info["pred_ac"] = pred_ac.cpu().detach()
        info["GT_ac"] = ac.cpu()
        diff = torch.sum(torch.abs(diff), axis=0).cpu()
        for i in range(diff.shape[0]):
            info['action'+ str(i) + '_L1loss'] = diff[i].mean().item()

        if train:
            # update the actor
            self._actor_optim.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
            sync_grads(self._actor)
            self._actor_optim.step()

        return mpi_average(info)
