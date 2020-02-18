import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dataset import ReplayBuffer, RandomSampler
from rl.base_agent import BaseAgent
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor


class DDPGAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic):
        super().__init__(config, ob_space)
        self._config = config

        # build up networks
        self._actor = actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = critic(config, ob_space, ac_space)

        # build up target networks
        self._actor_target = actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic_target = critic(config, ob_space, ac_space)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic_target.load_state_dict(self._critic.state_dict())
        self._network_cuda(self._config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)

        sampler = RandomSampler()
        buffer_keys = ['ob', 'ac', 'done', 'rew']
        self._buffer = ReplayBuffer(buffer_keys,
                                    config.buffer_size,
                                    sampler.sample_func)
        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a DDPG agent')
            logger.info('The actor has %d parameters', count_parameters(self._actor))
            logger.info('The critic has %d parameters', count_parameters(self._critic))

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic_state_dict': self._critic.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic_optim_state_dict': self._critic_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._critic.load_state_dict(ckpt['critic_state_dict'])
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic_target.load_state_dict(self._critic.state_dict())
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])

        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)


    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic.to(device)
        self._actor_target.to(device)
        self._critic_target.to(device)


    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)

    def train(self):
        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)

        self._soft_update_target_network(self._actor_target, self._actor, self._config.polyak)
        self._soft_update_target_network(self._critic_target, self._critic, self._config.polyak)

        train_info.update({
            'actor_grad_norm': compute_gradient_norm(self._actor),
            'actor_weight_norm': compute_weight_norm(self._actor),
            'critic_grad_norm': compute_gradient_norm(self._critic),
            'critic_weight_norm': compute_weight_norm(self._critic),
        })
        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process the observation
        o, o_next = transitions['ob'], transitions['ob_next']
        o = self.normalize(o)
        o_next = self.normalize(o_next)
        bs = len(transitions['done'])

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        done = _to_tensor(transitions['done']).reshape(bs, 1)
        # rew = reward from environment (e.g., collision penalty, interaction bonus)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, _ = self._actor_target(o_next)
            q_next_value = self._critic_target(o_next, actions_next)
            target_q_value = rew + (1 - done) * self._config.discount_factor * q_next_value
            ## clip the q value
            clip_return = 10 / (1 - self._config.discount_factor)
            target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        real_q_value = self._critic(o, ac)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real_q'] = real_q_value.min().cpu().item()
        info['real_q'] = real_q_value.mean().cpu().item()
        info['critic_loss'] = critic_loss.cpu().item()

        # the actor loss
        actions_real, _ = self._actor(o)
        actor_loss = -self._critic(o, actions_real).mean()
        info['actor_loss'] = actor_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic)
        self._critic_optim.step()

        # include info from policy
        info.update(self._actor.info)

        return mpi_average(info)