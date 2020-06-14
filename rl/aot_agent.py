from copy import deepcopy

import numpy as np
from torch.optim import Adam

from rl.base_agent import BaseAgent
from rl.dataset import ReplayBuffer
from rl.policies.utils import MLP
from util.mpi import mpi_average
from util.pytorch import (compute_gradient_norm, compute_weight_norm,
                          sync_grads, sync_networks, to_tensor)


class AoTAgent(BaseAgent):
    """
    creates an AoT classifier.
    preprocess_ob_func: gets the reversible portion of the observation. For example, we may want to ignore the agent dimensions if we are focused on the object.
    """

    def __init__(
        self, config, goal_space, dataset: ReplayBuffer, preprocess_ob_func=lambda x: x
    ):
        self._config = deepcopy(config)
        self._config.ob_norm = False
        self._dataset = dataset
        input_dim = goal_space[0]
        # TODO: replace with ensemble
        self.aot = MLP(config, input_dim, 1, [config.aot_hid_size] * 2).to(
            config.device
        )
        self._aot_optim = Adam(
            self.aot.parameters(),
            lr=config.lr_aot,
            weight_decay=config.aot_weight_decay,
        )
        self._reg_coeff = config.aot_reg_coeff
        self._preprocess = preprocess_ob_func

    def train(self) -> dict:
        """
        Sample X trajectories and Y timestep pairs per trajectory for a total batch size
        XY. Repeat this Z times.
        """
        train_info = {}
        num_eps = self._config.aot_num_episodes
        num_timepairs = self._config.aot_num_timepairs
        for i in range(self._config.aot_num_batches):
            transitions = self._get_time_pairs(self._dataset, num_eps, num_timepairs)
            train_info = self._update_network(transitions)
        train_info.update(
            {
                "aot_grad_norm": np.mean(compute_gradient_norm(self.aot)),
                "aot_weight_norm": np.mean(compute_weight_norm(self.aot)),
            }
        )
        return train_info

    def _update_network(self, transitions):
        """
        Transitions is (N,2) batch of timesteps
        """
        info = {}
        def _to_tensor(x):
            return to_tensor(x, self._config.device)

        transitions = _to_tensor(transitions)
        t = self.aot(transitions[:, :, 0])
        t1 = self.aot(transitions[:, :, 1])
        diff = t - t1  # (N, 1)
        aot_loss = diff.mean()
        aot_regularizer = diff.pow(2).mean() * self._reg_coeff
        loss = aot_loss + aot_regularizer
        info["aot_loss"] = aot_loss.cpu().item()
        info["aot_regularizer"] = aot_regularizer.cpu().item()
        info["aot_total_loss"] = loss.cpu().item()

        self._aot_optim.zero_grad()
        loss.backward()
        sync_grads(self.aot)
        self._aot_optim.step()

        return mpi_average(info)

    def _get_time_pairs(self, buffer, num_eps, num_timepairs) -> list:
        """Returns a list of (s, s') pairs from buffer"""
        data = buffer._buffer["ob"]
        num_episodes = len(data)
        sampled_eps_idx = np.random.randint(0, num_episodes, num_eps)
        episodes = [data[i] for i in sampled_eps_idx]
        t, t1 = [], []
        for ep in episodes:
            num_samples = min(len(ep) - 1, num_timepairs)  # don't oversample
            sampled_t = np.random.randint(0, len(ep) - 1, num_samples)
            for i in sampled_t:
                t.append(self._preprocess(ep[i]))
                t1.append(self._preprocess(ep[i + 1]))
        # should be size (N, ob_space, 2)
        all_pairs = np.stack([t, t1], axis=-1)
        return all_pairs

    def sync_networks(self):
        sync_networks(self.aot)


def test_get_time_pairs():
    # test agent training
    from config import create_parser
    from env import PegInsertionEnv
    from rl.dataset import RandomSampler
    from rl.rollouts import Rollout
    import torch

    parser = create_parser("PegInsertionEnv")
    config, _ = parser.parse_known_args()
    config.device = torch.device("cpu")
    env = PegInsertionEnv(config)
    sampler = RandomSampler()
    rb = ReplayBuffer(["ob", "ac"], 100, sampler.sample_func)
    aot = AoTAgent(config, env.goal_space, rb, env.get_goal)

    # generate rollouts of differing episode lengths
    for i in range(1, 6):
        r = Rollout()
        for j in range(i + 1):
            ob = {k: np.ones(v) * j for k, v in env.observation_space.items()}
            r.add({"ob": ob, "ac": 0})
        r.add({"ob": ob, "done": True})
        rb.store_episode(r.get())

    results = aot._get_time_pairs(rb, 5, 2)
    assert len(results) == 10, f"results is size {len(results)}"
    print("Get Time Pairs works")


def test_aot_train():
    # test agent training
    from config import create_parser
    from env import PegInsertionEnv
    from rl.dataset import RandomSampler
    from rl.rollouts import Rollout
    import torch

    parser = create_parser("PegInsertionEnv")
    config, _ = parser.parse_known_args()
    config.aot_num_episodes = 2
    config.aot_num_timepairs = 2
    config.aot_num_batches = 10
    config.aot_hid_size = 10
    config.device = torch.device("cpu")

    env = PegInsertionEnv(config)
    sampler = RandomSampler()
    rb = ReplayBuffer(["ob", "ac"], 100, sampler.sample_func)
    aot = AoTAgent(config, env.goal_space, rb, env.get_goal)

    # generate rollouts of differing episode lengths
    for i in range(1, 10):
        r = Rollout()
        for j in range(i + 1):
            ob = {k: np.ones(v) * j for k, v in env.observation_space.items()}
            r.add({"ob": ob, "ac": 0})
        r.add({"ob": ob, "done": True})
        rb.store_episode(r.get())

    train_info = aot.train()
    print("AoT train works")


if __name__ == "__main__":

    # test_get_time_pairs()
    test_aot_train()
