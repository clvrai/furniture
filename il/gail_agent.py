from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions

from rl.dataset import ReplayBuffer, RandomSampler
from il.bc_dataset import ILDataset
from rl.base_agent import BaseAgent
from rl.policies.discriminator import Discriminator
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, \
    obs2tensor, to_tensor


class GAILAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, actor, critic):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        # build up networks
        self._actor = actor(config, ob_space, ac_space, config.tanh_policy)
        self._old_actor = actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = critic(config, ob_space)
        self._discriminator = Discriminator(self._config, self._ob_space, self._ac_space)
        self._discriminator_loss = nn.BCEWithLogitsLoss()
        self._network_cuda(config.device)

        # build optimizers
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)
        self._discriminator_optim = optim.Adam(self._discriminator.parameters(), lr=config.lr_bc)

        # expert dataset
        self._dataset = ILDataset(config.demo_path)
        self._data_loader = torch.utils.data.DataLoader(self._dataset,
                                                        batch_size=self._config.batch_size,
                                                        shuffle=True)
        self._data_iter = iter(self._data_loader)

        # policy dataset
        sampler = RandomSampler()
        self._buffer = ReplayBuffer(['ob', 'ac', 'done', 'rew', 'ret', 'adv', 'ac_before_activation'],
                                    config.buffer_size,
                                    sampler.sample_func)

        self._log_creation()

    def predict_reward(self, ob, ac):
        ob = self.normalize(ob)
        with torch.no_grad():
            reward = self._discriminator.predict_reward(ob, ac)
        return reward.cpu().item()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a BC agent')
            logger.info('The actor has %d parameters', count_parameters(self._actor))

    def store_episode(self, rollouts):
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def _compute_gae(self, rollouts):
        T = len(rollouts['done'])
        ob = rollouts['ob']
        ob = self.normalize(ob)
        ob = obs2tensor(ob, self._config.device)
        vpred = self._critic(ob).detach().cpu().numpy()[:,0]
        assert len(vpred) == T + 1

        done = rollouts['done']
        rew = rollouts['rew']
        adv = np.empty((T, ) , 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = rew[t] + self._config.discount_factor * vpred[t + 1] * nonterminal - vpred[t]
            adv[t] = lastgaelam = delta + self._config.discount_factor * self._config.gae_lambda * nonterminal * lastgaelam

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        rollouts['adv'] = ((adv - adv.mean()) / adv.std()).tolist()
        rollouts['ret'] = ret.tolist()

    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic_state_dict': self._critic.state_dict(),
            'discriminator_state_dict': self._discriminator.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic_optim_state_dict': self._critic_optim.state_dict(),
            'discriminator_optim_state_dict': self._discriminator_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        if 'critic_state_dict' not in ckpt:
            logger.warn("Critic cannot be found in ckpt")
        else:
            self._critic.load_state_dict(ckpt['critic_state_dict'])
        if 'discriminator_state_dict' not in ckpt:
            logger.warn("Discriminator cannot be found in ckpt")
        else:
            self._discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        if 'critic_optim_state_dict' in ckpt:
            self._critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
        if 'discriminator_optim_state_dict' in ckpt:
            self._discriminator_optim.load_state_dict(ckpt['discriminator_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)
        optimizer_cuda(self._discriminator_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._old_actor.to(device)
        self._critic.to(device)
        self._discriminator.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._old_actor)
        sync_networks(self._critic)
        sync_networks(self._discriminator)

    def train(self):
        self._soft_update_target_network(self._old_actor, self._actor, 0.0)
        batch_size = self._config.batch_size

        for _ in range(self._config.num_batches):
            policy_data = self._buffer.sample(batch_size)
            train_info = self._update_policy(policy_data)

            try:
                expert_data = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self._data_loader)
                expert_data = next(self._data_iter)
            train_info2 = self._update_discriminator(policy_data, expert_data)
            train_info.update(train_info2)

        self._buffer.clear()

        train_info.update({
            'actor_grad_norm': compute_gradient_norm(self._actor),
            'actor_weight_norm': compute_weight_norm(self._actor),
            'critic_grad_norm': compute_gradient_norm(self._critic),
            'critic_weight_norm': compute_weight_norm(self._critic),
        })
        return train_info

    def _update_discriminator(self, policy_data, expert_data):
        info = {}

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        # pre-process observations
        p_o = policy_data['ob']
        p_o = self.normalize(p_o)

        p_bs = len(policy_data['ac'])
        p_o = _to_tensor(p_o)
        p_ac = _to_tensor(policy_data['ac'])

        e_o = expert_data['ob']
        e_o = self.normalize(e_o)

        e_bs = len(expert_data['ac'])
        e_o = _to_tensor(e_o)
        e_ac = _to_tensor(expert_data['ac'])

        p_logit = self._discriminator(p_o, p_ac)
        e_logit = self._discriminator(e_o, e_ac)

        p_output = torch.sigmoid(p_logit)
        e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(p_logit, torch.zeros_like(p_logit).to(self._config.device))
        e_loss = self._discriminator_loss(e_logit, torch.ones_like(e_logit).to(self._config.device))

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits).entropy().mean()
        entropy_loss = -self._config.gail_entropy_loss_coeff * entropy

        gail_loss = p_loss + e_loss + entropy_loss

        # update the discriminator
        self._discriminator.zero_grad()
        gail_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._discriminator)
        self._discriminator_optim.step()

        info['gail_policy_output'] = p_output.mean().detach().cpu().item()
        info['gail_expert_output'] = e_output.mean().detach().cpu().item()
        info['gail_entropy'] = entropy.detach().cpu().item()
        info['gail_policy_loss'] = p_loss.detach().cpu().item()
        info['gail_expert_loss'] = e_loss.detach().cpu().item()
        info['gail_entropy_loss'] = entropy_loss.detach().cpu().item()

        return mpi_average(info)

    def _update_policy(self, transitions):
        info = {}

        # pre-process observations
        o = transitions['ob']
        o = self.normalize(o)

        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions['ac'])
        a_z = _to_tensor(transitions['ac_before_activation'])
        ret = _to_tensor(transitions['ret']).reshape(bs, 1)
        adv = _to_tensor(transitions['adv']).reshape(bs, 1)

        log_pi, ent = self._actor.act_log(o, a_z)
        old_log_pi, _ = self._old_actor.act_log(o, a_z)
        if old_log_pi.min() < -100:
            import ipdb; ipdb.set_trace()

        # the actor loss
        entropy_loss = self._config.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self._config.clip_param,
                            1.0 + self._config.clip_param) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        if not np.isfinite(ratio.cpu().detach()).all() or not np.isfinite(adv.cpu().detach()).all():
            import ipdb; ipdb.set_trace()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # the q loss
        value_pred = self._critic(o)
        value_loss = self._config.value_loss_coeff * (ret - value_pred).pow(2).mean()

        info['value_target'] = ret.mean().cpu().item()
        info['value_predicted'] = value_pred.mean().cpu().item()
        info['value_loss'] = value_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic)
        self._critic_optim.step()

        # include info from policy
        info.update(self._actor.info)

        return mpi_average(info)
