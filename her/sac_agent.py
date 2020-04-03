# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from her.her_policy import CNN, MLP
from her.dataset import ReplayBuffer, HERSampler
from her.distributions import FixedNormal
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
)
from time import time


class Actor(nn.Module):
    def __init__(self, config, ob_space, goal_space, ac_space):
        super().__init__()
        self._config = config
        self._activation_fn = getattr(F, config.activation)

        # observation
        if config.visual_ob:
            self._cnn_encoder = CNN(config)
            input_dim = self._cnn_encoder.output_size
        else:
            input_dim = np.prod(ob_space["object_ob"])
        input_dim += np.prod(ob_space["robot_ob"]) if config.robot_ob else 0

        # goal
        input_dim += np.prod(goal_space)

        self.fc = MLP(config, input_dim, config.hid_size, [config.hid_size])
        self.fc_mean = MLP(config, config.hid_size, ac_space)
        self.fc_log_std = MLP(config, config.hid_size, ac_space)

    def forward(self, ob, g, deterministic=False, return_log_prob=False):
        inp = []

        if self._config.robot_ob:
            r = ob["robot_ob"]
            if len(r.shape) == 1:
                r = r.unsqueeze(0)
            inp.append(r)

        if self._config.visual_ob:
            o = ob["normal"]
            if len(o.shape) == 3:
                o = o.unsqueeze(0)
            o = o.permute(0, 3, 1, 2)
            o = self._activation_fn(self._cnn_encoder(o))
        else:
            o = ob["object_ob"]
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
        inp.append(o)

        if len(g.shape) == 1:
            g = g.unsqueeze(0)
        inp.append(g)

        out = self._activation_fn(self.fc(torch.cat(inp, dim=-1)))
        out = torch.reshape(out, (out.shape[0], -1))
        mean = self.fc_mean(out)
        log_std = self.fc_log_std(out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            z = mean
        else:
            dist = FixedNormal(mean, std)
            z = dist.rsample()
            if return_log_prob:
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.0) - z - F.softplus(-2.0 * z)).sum(
                    dim=-1, keepdim=True
                )
                log_prob = dist.log_probs(z) - log_det_jacobian
                return action, log_prob

        return torch.tanh(z)

    def _to_tensor(self, x):
        if isinstance(x, dict):
            return {
                k: torch.tensor(v).to(self._config.device, dtype=torch.float32)
                for k, v in x.items()
            }
        return torch.tensor(x, dtype=torch.float32).to(self._config.device)

    def act(self, ob, g, is_train=True):
        ob = self._to_tensor(ob)
        g = self._to_tensor(g)

        ac = self.forward(ob, g, deterministic=not is_train)
        ac = ac.detach().cpu().numpy().squeeze()
        return ac


class Critic(nn.Module):
    def __init__(self, config, ob_space, goal_space, ac_space):
        super().__init__()
        self._config = config
        self._activation_fn = getattr(F, config.activation)

        if config.visual_ob:
            self._cnn_encoder = CNN(config)
            input_dim = self._cnn_encoder.output_size
        else:
            input_dim = np.prod(ob_space["object_ob"])

        input_dim += np.prod(ob_space["robot_ob"]) if config.robot_ob else 0
        input_dim += np.prod(goal_space)
        input_dim += ac_space

        self.fc = MLP(config, input_dim, 1, [config.hid_size] * 2)

    def forward(self, ob, g, ac):
        inp = []

        if self._config.robot_ob:
            r = ob["robot_ob"]
            if len(r.shape) == 1:
                r = r.unsqueeze(0)
            inp.append(r)

        if self._config.visual_ob:
            o = ob["normal"]
            if len(o.shape) == 3:
                o = o.unsqueeze(0)
            o = o.permute(0, 3, 1, 2)
            o = self._activation_fn(self._cnn_encoder(o))
        else:
            o = ob["object_ob"]
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
        inp.append(o)

        if len(g.shape) == 1:
            g = g.unsqueeze(0)
        inp.append(g)

        if len(ac.shape) == 1:
            ac = ac.unsqueeze(0)
        inp.append(ac)

        out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out


class SACAgent(object):
    def __init__(self, config, env, ob_space, goal_space, ac_space, ob_norm, g_norm):
        self._config = config
        self._env = env

        self._ob_norm = ob_norm
        self._g_norm = g_norm

        self._target_entropy = -ac_space
        self._log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor)

        # no control penalty for SAC
        config.action_l2 = 0.0

        # build up networks
        self._actor = Actor(config, ob_space, goal_space, ac_space)
        self._critic1 = Critic(config, ob_space, goal_space, ac_space)
        self._critic2 = Critic(config, ob_space, goal_space, ac_space)

        # build up target networks
        self._critic1_target = Critic(config, ob_space, goal_space, ac_space)
        self._critic2_target = Critic(config, ob_space, goal_space, ac_space)
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

        her_module = HERSampler(
            config.replay_strategy, config.replace_future, self._env.compute_reward
        )
        self._buffer = ReplayBuffer(
            config.buffer_size, her_module.sample_her_transitions
        )

        if config.is_chef:
            logger.info("Creating a SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info(
                "The critic1 has %d parameters", count_parameters(self._critic1)
            )
            logger.info(
                "The critic2 has %d parameters", count_parameters(self._critic2)
            )

    def act(self, ob, g, is_train=True):
        return self._actor.act(ob, g, is_train)

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
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic1_optim.load_state_dict(ckpt["critic1_optim_state_dict"])
        self._critic2_optim.load_state_dict(ckpt["critic2_optim_state_dict"])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def replay_buffer(self):
        return self._buffer.state_dict()

    def load_replay_buffer(self, state_dict):
        self._buffer.load_state_dict(state_dict)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def _soft_update_target_network(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic1)
        sync_networks(self._critic2)

    def train(self):
        start = time()
        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)
            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.polyak
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.polyak
            )
        end = time() - start

        train_info.update(
            {
                "actor_grad_norm": compute_gradient_norm(self._actor),
                "actor_weight_norm": compute_weight_norm(self._actor),
                "critic1_grad_norm": compute_gradient_norm(self._critic1),
                "critic2_grad_norm": compute_gradient_norm(self._critic2),
                "critic1_weight_norm": compute_weight_norm(self._critic1),
                "critic2_weight_norm": compute_weight_norm(self._critic2),
            }
        )
        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process the observation and goal
        o, o_next = transitions["ob"], transitions["ob_next"]
        g = transitions["g"]

        bs = len(g)

        if self._config.ob_norm:
            o = self._ob_norm.normalize(o)
            o_next = self._ob_norm.normalize(o_next)
            g = self._g_norm.normalize(g)

        # transfer them into the tensor
        def _to_tensor(x):
            if isinstance(x, dict):
                ret = {
                    k: torch.tensor(v, dtype=torch.float32).to(self._config.device)
                    for k, v in x.items()
                }
            else:
                ret = torch.tensor(x, dtype=torch.float32).to(self._config.device)
            return ret

        o = _to_tensor(o)
        g = g_next = _to_tensor(g)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        # r = 0 if it reaches 'g', otherwise -1
        r = _to_tensor(transitions["r"]).reshape(bs, 1)
        done = 1.0 + r
        # rew = reward from environment (e.g., collision penalty, interaction bonus)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        # update alpha
        actions_real, log_pi = self._actor(o, g, return_log_prob=True)
        alpha_loss = -(
            self._log_alpha * (log_pi + self._target_entropy).detach()
        ).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(
            self._critic1(o, g, actions_real), self._critic2(o, g, actions_real)
        ).mean()
        control_loss = self._config.action_l2 * actions_real.pow(2).mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        info["control_loss"] = control_loss.cpu().item()
        actor_loss += entropy_loss + control_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self._actor(
                o_next, g_next, return_log_prob=True
            )
            q_next_value1 = self._critic1_target(o_next, g_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, g_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (r + rew) * self._config.reward_scale + (
                1 - done
            ) * self._config.gamma * q_next_value
            target_q_value = target_q_value.detach()
            ## clip the q value
            clip_return = 1 / (1 - self._config.gamma)
            ## in HER target_q_value is always negative
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value1 = self._critic1(o, g, ac)
        real_q_value2 = self._critic2(o, g, ac)
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

        return mpi_average(info)
