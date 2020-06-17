from copy import deepcopy

import numpy as np
import torch
from torch.nn.functional import softmax, softmin
from torch.optim import Adam

from rl.base_agent import BaseAgent
from rl.dataset import ReplayBuffer
from rl.policies.utils import MLP, Ensemble
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import (compute_gradient_norm, compute_weight_norm,
                          count_parameters, sync_grads, sync_networks,
                          to_tensor)


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
        self._device = config.device
        input_dim = goal_space[0]

        def create_mlp():
            return MLP(config, input_dim, 1, [config.aot_hid_size] * 2)
        if config.aot_ensemble is not None:
            self._aot = Ensemble(create_mlp, config.aot_ensemble)
        else:
            self._aot = create_mlp()
        self._network_cuda(self._device)
        self._aot_optim = Adam(
            self._aot.parameters(),
            lr=config.lr_aot,
            weight_decay=config.aot_weight_decay,
        )
        self._reg_coeff = config.aot_reg_coeff
        self._preprocess_ob_func = preprocess_ob_func
        self._aot_rew_coeff = config.aot_rew_coeff
        self._ensemble = config.aot_ensemble is not None
        self._ensemble_sampler = config.ensemble_sampler
        self._var_coeff = config.var_coeff

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating an AoT agent")
            logger.info("The AoT has %d parameters", count_parameters(self._aot))

    def act(self, ob, is_train=True, return_info=False):
        """
        return_info: returns supplemental info about ensemble
        Ob should either be a single ob dict or a list of ob dicts and respecitively
        will get converted to a (E, ob_space) or (E, N, ob_space) tensor.
        Out will be (E, 1) or (E, N, 1)
        Sampler will return 1 sample for (E,1) or N sample for (E,N,1)
        For each ob in N obs, return one idx between [0, E)
        """
        self._aot.train(is_train)
        single_ob = isinstance(ob, dict)
        ob = to_tensor(self._preprocess(ob), self._device)
        out = self._aot(ob)
        if not self._ensemble:
            return out
        info = {}
        N = out.shape[1]
        # choose sampling strategy for ensemble. Output is (E, 1) or (E,N,1)
        if self._ensemble_sampler == "min":
            sample = out.min(dim=0).values
        elif self._ensemble_sampler == "max":
            sample = out.max(dim=0).values
        elif self._ensemble_sampler == "mean":
            sample = out.mean(dim=0).detach()
        elif self._ensemble_sampler == "median":
            sample = out.median(dim=0).values
        elif self._ensemble_sampler in ["softmin", "softmax"]:
            soft = softmin if self._ensemble_sampler == "softmin" else softmax
            probs = soft(out, dim=0).squeeze()
            if single_ob:
                min_idx = torch.multinomial(probs, 1).flatten()
                sample = out[min_idx].squeeze(dim=-1)
            else:
                sample = []
                for i in range(N):
                    p = probs[:, i].flatten()
                    min_idx = torch.multinomial(p, 1).flatten()
                    sample.append(out[min_idx, i].squeeze(dim=-1))
                sample = torch.stack(sample)
        elif self._ensemble_sampler == "meanvar":
            mean = out.mean(dim=0)
            var = out.var(dim=0)
            sample = mean - self._var_coeff * var
        else:
            raise NotImplementedError(self._ensemble_sampler)

        sample = sample.detach()
        if return_info:
            mean = out.mean(dim=0).detach()
            var = out.var(dim=0).detach()
            info.update({"out": out, "mean": mean, "var": var})
            return sample, info
        return sample

    @torch.no_grad()
    def rew(self, ob, ob_next):
        """
        Reward function using AoT. Since AoT increases for s1, s2, s3, we want to
        find a next state whose AoT is lower than our current one. Hence rew is
        AoT(s) - AoT(s').
        """
        rew = self.act(ob, False) - self.act(ob_next, False)
        rew = rew.cpu().numpy().flatten() * self._aot_rew_coeff
        return rew

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
                "aot_grad_norm": np.mean(compute_gradient_norm(self._aot)),
                "aot_weight_norm": np.mean(compute_weight_norm(self._aot)),
            }
        )
        return train_info

    def _update_network(self, transitions):
        """
        Transitions is (N,2) batch of timesteps
        """
        self._aot.train()
        info = {}
        transitions = to_tensor(transitions, self._config.device)
        t = self._aot(transitions[:, :, 0])
        t1 = self._aot(transitions[:, :, 1])
        diff = t - t1  # (N, 1) or (E,N,1) if Ensemble
        aot_loss = diff.mean()  # We can average losses even if E since indep.
        aot_regularizer = diff.pow(2).mean() * self._reg_coeff  # same here
        loss = aot_loss + aot_regularizer
        info["aot_loss"] = aot_loss.cpu().item()
        info["aot_regularizer"] = aot_regularizer.cpu().item()
        info["aot_total_loss"] = loss.cpu().item()

        self._aot_optim.zero_grad()
        loss.backward()
        sync_grads(self._aot)
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
            # num_samples = min(len(ep) - 1, num_timepairs)  # don't oversample
            sampled_t = np.random.randint(0, len(ep) - 1, num_timepairs)
            for i in sampled_t:
                t.append(self._preprocess(ep[i]))
                t1.append(self._preprocess(ep[i + 1]))
        # should be size (N, ob_space, 2)
        all_pairs = np.stack([t, t1], axis=-1)
        return all_pairs

    def sync_networks(self):
        sync_networks(self._aot)

    def _preprocess(self, ob):
        if isinstance(ob, dict):
            return self._preprocess_ob_func(ob)
        elif isinstance(ob, list):
            return np.stack([self._preprocess_ob_func(o) for o in ob])
        else:
            raise NotImplementedError(type(ob))

    def state_dict(self):
        return {
            "aot_state_dict": self._aot.state_dict(),
            "aot_optim_state_dict": self._aot_optim.state_dict()
        }

    def load_state_dict(self, ckpt):
        self._aot.load_state_dict(ckpt["aot_state_dict"])
        self._aot_optim.load_state_dict(ckpt["aot_optim_state_dict"])
        self._network_cuda(self._config.device)

    def _network_cuda(self, device):
        self._aot.to(device)


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
    config.is_chef = True

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
    config.is_chef = True

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

    _ = aot.train()
    print("AoT train works")


def test_act():
    # test agent training
    from config import create_parser
    from env import PegInsertionEnv
    from rl.dataset import RandomSampler
    import torch

    parser = create_parser("PegInsertionEnv")
    config, _ = parser.parse_known_args()
    config.aot_num_episodes = 2
    config.aot_num_timepairs = 2
    config.aot_num_batches = 10
    config.aot_hid_size = 10
    config.device = torch.device("cpu")
    config.is_chef = True

    env = PegInsertionEnv(config)
    sampler = RandomSampler()
    rb = ReplayBuffer(["ob", "ac"], 100, sampler.sample_func)
    aot = AoTAgent(config, env.goal_space, rb, env.get_goal)

    # test 1 ob
    ob = {k: np.ones(v) * 5 for k, v in env.observation_space.items()}
    output = aot.act(ob)
    assert list(output.shape) == [1], output.shape

    # test batched ob
    batch_ob = [
        {k: np.ones(v) * 5 for k, v in env.observation_space.items()} for i in range(5)
    ]
    output = aot.act(batch_ob)
    assert list(output.shape) == [5, 1], output.shape
    print("AoT act works")


def test_ensemble_train():
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
    config.is_chef = True

    config.aot_ensemble = 2

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

    _ = aot.train()
    print("AoT ensemble train works")


def test_ensemble_act():
    # test agent training
    from config import create_parser
    from env import PegInsertionEnv
    from rl.dataset import RandomSampler
    import torch

    parser = create_parser("PegInsertionEnv")
    config, _ = parser.parse_known_args()
    config.aot_num_episodes = 2
    config.aot_num_timepairs = 2
    config.aot_num_batches = 10
    config.aot_hid_size = 10
    config.device = torch.device("cpu")
    config.is_chef = True

    config.aot_ensemble = 2

    env = PegInsertionEnv(config)
    sampler = RandomSampler()
    rb = ReplayBuffer(["ob", "ac"], 100, sampler.sample_func)
    aot = AoTAgent(config, env.goal_space, rb, env.get_goal)

    choices = ["min", "mean", "max", "median", "meanvar", "softmin"]
    # test 1 ob
    ob = {k: np.ones(v) * 5 for k, v in env.observation_space.items()}
    output = aot.act(ob)
    for s in choices:
        aot._ensemble_sampler = s
        output = aot.act(ob)
        assert list(output.shape) == [1], f"{s}:{output.shape}"
    # test batched ob
    batch_ob = [
        {k: np.ones(v) * 5 for k, v in env.observation_space.items()} for i in range(5)
    ]
    for s in choices:
        aot._ensemble_sampler = s
        output = aot.act(batch_ob)
        assert list(output.shape) == [5, 1], f"{s}:{output.shape}"
    print("AoT ensemble act works")


if __name__ == "__main__":
    # test_get_time_pairs()
    # test_aot_train()
    # test_act()
    # test_ensemble_train()
    test_ensemble_act()
