# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py


from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence

from rl.base_agent import BaseAgent
from rl.dataset import LearnedRewardReplayBuffer, RandomSampler, ReplayBuffer
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import (compute_gradient_norm, compute_weight_norm,
                          count_parameters, optimizer_cuda, sync_grads,
                          sync_networks, to_tensor)


class SACAgent(BaseAgent):
    def __init__(
        self, config, ob_space, ac_space, actor, critic, reset=False, rew=None
    ):

        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._target_entropy = -ac_space.size
        self._log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor)

        # build up networks
        self._build_actor(actor)
        self._critic1 = critic(config, ob_space, ac_space)
        self._critic2 = critic(config, ob_space, ac_space)

        # build up target networks
        self._critic1_target = critic(config, ob_space, ac_space)
        self._critic2_target = critic(config, ob_space, ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic1_optim = optim.Adam(
            self._critic1.parameters(), lr=config.lr_critic
        )
        self._critic2_optim = optim.Adam(
            self._critic2.parameters(), lr=config.lr_critic
        )

        sampler = RandomSampler()
        buffer_keys = ["ob", "ac", "done", "rew"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )
        self._reset_agent = reset
        if config.use_aot and reset:
            assert rew is not None
            buffer_keys.remove("rew")  # replace rew with env_rew
            buffer_keys.append("env_rew")
            self._buffer = LearnedRewardReplayBuffer(
                buffer_keys, config.buffer_size, sampler.sample_func, rew
            )

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info(
                "The critic1 has %d parameters", count_parameters(self._critic1)
            )
            logger.info(
                "The critic2 has %d parameters", count_parameters(self._critic2)
            )

    def _build_actor(self, actor):
        self._actor = actor(
            self._config, self._ob_space, self._ac_space, self._config.tanh_policy
        )

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            "log_alpha": self._log_alpha.cpu().detach().numpy(),
            "actor_state_dict": self._actor.state_dict(),
            "critic1_state_dict": self._critic1.state_dict(),
            "critic2_state_dict": self._critic2.state_dict(),
            "alpha_optim_state_dict": self._alpha_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic1_optim_state_dict": self._critic1_optim.state_dict(),
            "critic2_optim_state_dict": self._critic2_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._config.device
        )
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic1.load_state_dict(ckpt["critic1_state_dict"])
        self._critic2.load_state_dict(ckpt["critic2_state_dict"])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic1_optim.load_state_dict(ckpt["critic1_optim_state_dict"])
        self._critic2_optim.load_state_dict(ckpt["critic2_optim_state_dict"])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic1)
        sync_networks(self._critic2)

    def train(self):
        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )

        train_info.update(
            {
                "actor_grad_norm": np.mean(compute_gradient_norm(self._actor)),
                "actor_weight_norm": np.mean(compute_weight_norm(self._actor)),
                "critic1_grad_norm": compute_gradient_norm(self._critic1),
                "critic2_grad_norm": compute_gradient_norm(self._critic2),
                "critic1_weight_norm": compute_weight_norm(self._critic1),
                "critic2_weight_norm": compute_weight_norm(self._critic2),
            }
        )
        return train_info

    @torch.no_grad()
    def is_safe_action(self, ob, ac):
        # TODO: handle batch and single action modes
        ob = to_tensor(self.normalize(ob), self._config.device)
        ac = to_tensor(ac, self._config.device)
        values = torch.min(self._reset_critic1(ob, ac), self._reset_critic2(ob, ac))
        safe = values > self._config.safety_threshold
        return safe.item()

    @torch.no_grad()
    def safe_act(self, ob, is_train=True):
        """
        Act safely w.r.t a safety threshold
        """
        ob = self.normalize(ob)
        obs = deepcopy(ob)
        # sample other actions from the forward policy
        N = self._config.num_safety_actions
        threshold = self._config.safety_threshold
        # duplicate the observaons
        for k, v in obs.items():
            obs[k] = N * [v]
        obs = to_tensor(obs, self._config.device)
        actions, activations = self.act(obs, is_train)
        reset_values = torch.min(
            self._reset_critic1(obs, actions), self._reset_critic2(obs, actions)
        )
        over_threshold = torch.flatten(reset_values > threshold)
        # choose uniformly an acceptable action
        safe_actions = actions["default"][over_threshold]
        safe_activations = activations[over_threshold]
        if len(safe_actions) > 0:
            choice = np.random.randint(len(safe_actions))
            out = safe_actions[choice], safe_activations[choice]
        else:
            choice = np.random.randint(len(actions))
            out = actions[choice], activations[choice]
        return out

    def act_log(self, ob):
        return self._actor.act_log(ob)

    def set_reset_critic(self, critic1, critic2):
        self._reset_critic1 = critic1
        self._reset_critic2 = critic2

    def set_forward_actor(self, actor):
        self._forward_actor = actor

    def _update_network(self, transitions):
        info = {}

        # pre-process observations
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)

        bs = len(transitions["done"])

        def _to_tensor(x):
            return to_tensor(x, self._config.device)

        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        done = _to_tensor(transitions["done"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        # update alpha
        actions_real, log_pi = self.act_log(o)
        alpha_loss = -(
            self._log_alpha.exp() * (log_pi + self._target_entropy).detach()
        ).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(
            self._critic1(o, actions_real), self._critic2(o, actions_real)
        ).mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss
        if self._config.reset_kl_penalty and self._reset_agent:
            reset_distributions = self._actor.act_dist(o)
            with torch.no_grad():
                forward_distributions = self._forward_actor.act_dist(o)
            f = forward_distributions.distributions["default"]
            r = reset_distributions.distributions["default"]
            kl_loss = self._config.kl_penalty_coeff * kl_divergence(f, r).mean().clamp_(
                0, 1000
            )
            info["kl_loss"] = kl_loss.cpu().item()
            # bigger the kl_d, the better
            actor_loss -= kl_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (
                rew * self._config.reward_scale
                + (1 - done) * self._config.discount_factor * q_next_value
            )
            target_q_value = target_q_value.detach()
            ## clip the q value
            clip_return = 10 / (1 - self._config.discount_factor)
            target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real1_q"] = real_q_value1.min().cpu().item()
        info["min_real2_q"] = real_q_value2.min().cpu().item()
        info["real1_q"] = real_q_value1.mean().cpu().item()
        info["real2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic2)
        self._critic2_optim.step()

        # include info from policy
        info.update(self._actor.info)
        return mpi_average(info)
