import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from her.distributions import Categorical, DiagGaussian

from her.her_policy import CNN, MLP
from her.dataset import MetaReplayBuffer, PrioritizedReplayBuffer
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


class MetaActor(nn.Module):
    def __init__(self, config, ob_space, goal_space, ac_space):
        super().__init__()
        self._config = config
        self._activation_fn = getattr(F, config.activation)
        self.eps = 1.0
        if self._config.meta_agent != "policy":
            return
        # observation
        if config.visual_ob:
            self._cnn_encoder = CNN(config)
            input_dim = self._cnn_encoder.output_size
        else:
            input_dim = np.prod(ob_space["object_ob"])
        input_dim += np.prod(ob_space["robot_ob"]) if config.robot_ob else 0

        # goal
        input_dim += np.prod(goal_space)

        self.fc = MLP(config, input_dim, 1, [config.hid_size] * 2)

    def forward(self, ob, g):
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
        inp = torch.cat(inp, dim=-1)

        if len(g.shape) == 2:
            g = g.unsqueeze(0)

        out = []
        decay = 1.0
        for i in range(self._config.meta_window):
            out.append(decay * self.fc(torch.cat([inp, g[:, i, :]], dim=-1)))
            decay *= self._config.meta_reward_decay

        out = torch.cat(out, dim=-1)

        return out

    def _to_tensor(self, x):
        if isinstance(x, dict):
            return {
                k: torch.tensor(v).to(self._config.device, dtype=torch.float32)
                for k, v in x.items()
            }
        return torch.tensor(x, dtype=torch.float32).to(self._config.device)

    def act(self, ob, g, demo_i, is_train=True):
        if len(g) - demo_i - 2 <= 0:
            return len(g) - demo_i - 2
        if self._config.meta_agent == "sequential":
            return 0
        elif self._config.meta_agent == "random":
            return np.random.randint(min(len(g) - demo_i - 1, self._config.meta_window))

        ob = self._to_tensor(ob)

        sampled_g = []
        for i in range(self._config.meta_window):
            if demo_i + i + 1 >= len(g):
                sampled_g.append(g[-1])
            else:
                sampled_g.append(g[demo_i + i + 1])

        sampled_g = self._to_tensor(np.stack(sampled_g))
        q_values = self.forward(ob, sampled_g)
        q_values = q_values.detach().cpu().numpy().squeeze()
        ac = np.argmax(q_values)
        if is_train and np.random.uniform() < self.eps:
            ac = np.random.randint(self._config.meta_window)

        return min(ac, len(g) - demo_i - 2)

    def update_eps(self):
        self.eps -= self._config.epsilon_decay
        self.eps = np.clip(self.eps, 0.1, 1.0)


class MetaAgent(object):
    def __init__(self, config, env, ob_space, goal_space, ob_norm, g_norm):
        self._config = config
        self._env = env

        self._ob_norm = ob_norm
        self._g_norm = g_norm

        # build up networks
        self._actor = MetaActor(config, ob_space, goal_space, config.meta_window)

        # build up target networks
        self._actor_target = MetaActor(config, ob_space, goal_space, config.meta_window)

        if self._config.meta_agent == "policy":
            self._actor_target.load_state_dict(self._actor.state_dict())
            self._network_cuda(self._config.device)

            self._actor_optim = optim.Adam(
                self._actor.parameters(), lr=self._config.lr_actor
            )

            if config.use_per:
                self._buffer = PrioritizedReplayBuffer(
                    config.buffer_size, config.meta_window
                )
            else:
                self._buffer = MetaReplayBuffer(config.buffer_size, config.meta_window)

        if config.is_chef:
            logger.info("Creating a meta agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))

    def act(self, ob, g, demo_i, is_train=True):
        return self._actor.act(ob, g, demo_i, is_train)

    def store_episode(self, rollouts):
        if self._config.meta_agent == "policy":
            self._buffer.store_episode(rollouts)

    def state_dict(self) -> dict:
        if self._config.meta_agent == "policy":
            return {
                "actor_eps": self._actor.eps,
                "actor_state_dict": self._actor.state_dict(),
                "actor_optim_state_dict": self._actor_optim.state_dict(),
            }
        return {}

    def load_state_dict(self, ckpt):
        if self._config.meta_agent == "policy":
            self._actor.load_state_dict(ckpt["actor_state_dict"])
            self._actor_target.load_state_dict(self._actor.state_dict())
            self._network_cuda(self._config.device)

            self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
            optimizer_cuda(self._actor_optim, self._config.device)

            self._actor.eps = ckpt["actor_eps"]

    def replay_buffer(self):
        if self._config.meta_agent == "policy":
            return self._buffer.state_dict()
        return {}

    def load_replay_buffer(self, state_dict):
        if self._config.meta_agent == "policy":
            self._buffer.load_state_dict(state_dict)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._actor_target.to(device)

    def _soft_update_target_network(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

    def sync_networks(self):
        if self._config.meta_agent == "policy":
            sync_networks(self._actor)

    def train(self):
        if self._config.meta_agent != "policy":
            return {}

        self._actor.update_eps()

        for _ in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions)
            self._soft_update_target_network(
                self._actor_target, self._actor, self._config.polyak
            )

        train_info.update(
            {
                "actor_eps": self._actor.eps,
                "actor_grad_norm": compute_gradient_norm(self._actor),
                "actor_weight_norm": compute_weight_norm(self._actor),
            }
        )
        return train_info

    def _update_network(self, transitions):
        info = {}

        # pre-process the observation and goal
        o, o_next = transitions["ob"], transitions["ob_next"]
        g, g_next = transitions["g"], transitions["g_next"]

        bs = len(g)

        if self._config.ob_norm:
            o = self._ob_norm.normalize(o)
            o_next = self._ob_norm.normalize(o_next)
            g_dim = g.shape
            g = self._g_norm.normalize(g.reshape(-1, g_dim[-1])).reshape(g_dim)
            g_next = self._g_norm.normalize(g_next.reshape(-1, g_dim[-1])).reshape(
                g_dim
            )

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
        o_next = _to_tensor(o_next)
        g = _to_tensor(g)
        g_next = _to_tensor(g_next)
        # ac = _to_tensor(transitions['ac'])
        ac = torch.eye(self._config.meta_window).to(self._config.device)[
            transitions["ac"]
        ]
        # normalize reward to make the total reward between [0, 1]
        r = _to_tensor(transitions["rew"]).reshape(
            bs, 1
        )  # / self._config.max_episode_steps
        done = _to_tensor(transitions["done"]).reshape(bs, 1)
        if self._config.use_per:
            weights = _to_tensor(transitions["weight"]).reshape(bs, 1)
        else:
            weights = torch.ones_like(r)

        # calculate the target Q value function
        if self._config.binary_q:
            target_q_value = r
        else:
            with torch.no_grad():
                q_next_value = self._actor_target(o_next, g_next)
                if self._config.meta_algo == "ddqn":
                    best_ac = self._actor(o_next, g_next).max(1, keepdim=True)[1]
                    q_next_value = q_next_value.detach().gather(1, best_ac)
                else:
                    q_next_value = q_next_value.detach().max(1, keepdim=True)[0]
                target_q_value = r + (1 - done) * self._config.gamma * q_next_value
                target_q_value = target_q_value.detach()

        # the q loss
        real_q_value = torch.sum(self._actor(o, g) * ac, dim=1, keepdim=True)
        critic_diff = (target_q_value - real_q_value).pow(2)
        critic_loss = (critic_diff * weights).mean()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real_q"] = real_q_value.min().cpu().item()
        info["real_q"] = real_q_value.mean().cpu().item()
        info["critic_loss"] = critic_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        # update prioritized experience replay
        if self._config.use_per:
            idx = transitions["indexes"]
            priority = critic_diff.detach().cpu().numpy() + self._config.per_eps
            self._buffer.update_priorities(idx, priority)

        return mpi_average(info)
