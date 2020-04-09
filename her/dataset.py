from collections import defaultdict
from time import time
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, sample_func):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = {"ob", "ag", "g", "ac", "rew"}
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])


class MetaReplayBuffer:
    def __init__(self, buffer_size, meta_window):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0

        # create the buffer to store info
        self._keys = {"ob", "ac", "rew", "done"}
        self._buffers = defaultdict(list)
        self._meta_window = meta_window

    def _add_value(self, k, v, idx):
        if self._current_size > self._size:
            self._buffers[k][idx] = v
        else:
            self._buffers[k].append(v)

    # store the episode
    def store_episode(self, rollout):
        demo_i = rollout["demo_i"][0]
        sampled_g = []
        for j in range(self._meta_window):
            if demo_i + j + 1 >= len(rollout["demo"]):
                sampled_g.append(rollout["demo"][-1])
            else:
                sampled_g.append(rollout["demo"][demo_i + j + 1])
        sampled_g = np.stack(sampled_g)

        for i in range(len(rollout["ac"])):
            idx = self._idx = (self._idx + 1) % self._size
            self._current_size += 1
            for k in self._keys:
                self._add_value(k, rollout[k][i], idx)
            self._add_value("ob_next", rollout["ob"][i + 1], idx)

            self._add_value("g", sampled_g, idx)

            demo_i = rollout["demo_i"][i + 1]
            sampled_g = []
            for j in range(self._meta_window):
                if demo_i + j + 1 >= len(rollout["demo"]):
                    sampled_g.append(rollout["demo"][-1])
                else:
                    sampled_g.append(rollout["demo"][demo_i + j + 1])
            sampled_g = np.stack(sampled_g)
            self._add_value("g_next", sampled_g, idx)

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # start = time()
        episode_batch = self._buffers
        rollout_batch_size = len(episode_batch["ac"])

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx] for episode_idx in episode_idxs
            ]
        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        # print('sample done: {} sec'.format(time() - start))
        return new_transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])


class HERSampler:
    def __init__(self, replay_strategy, replace_future, reward_func=None):
        self.replay_strategy = replay_strategy
        if self.replay_strategy == "future":
            self.future_p = replace_future
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        # start = time()
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob"][episode_idx][t + 1]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]
        transitions["r"] = np.zeros((batch_size,))

        # hindsight experience replay
        for i, (episode_idx, t) in enumerate(zip(episode_idxs, t_samples)):
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_t = np.random.randint(
                    t + 1, len(episode_batch["ac"][episode_idx]) + 1
                )
                future_ag = episode_batch["ag"][episode_idx][future_t]
                if (
                    self.reward_func(
                        episode_batch["ag"][episode_idx][t], future_ag, None
                    )
                    < 0
                ):
                    transitions["g"][i] = future_ag
            transitions["r"][i] = self.reward_func(
                episode_batch["ag"][episode_idx][t + 1], transitions["g"][i], None
            )

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)
        # print('sample done: {} sec'.format(time() - start))

        return new_transitions


# code from https://github.com/medipixel/rl_algorithms/blob/master/algorithms/common/buffer/priortized_replay_buffer.py
# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for algorithms.
- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

from util.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, meta_window, alpha=0.6):
        """Initialization.
        Args:
            buffer_size (int): size of replay buffer for experience
            alpha (float): alpha parameter for prioritized replay buffer
        """
        assert alpha >= 0
        self._size = buffer_size
        self._keys = {"ob", "ac", "rew", "done"}
        self._buffers = defaultdict(list)
        self._meta_window = meta_window
        self._alpha = alpha
        self._beta = 0.4

        self._idx = -1
        self._current_size = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self._size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def _add_value(self, k, v, idx):
        if self._current_size > self._size:
            self._buffers[k][idx] = v
        else:
            self._buffers[k].append(v)

    # store the episode
    def store_episode(self, rollout):
        demo_i = rollout["demo_i"][0]
        sampled_g = []
        for j in range(self._meta_window):
            if demo_i + j + 1 >= len(rollout["demo"]):
                sampled_g.append(rollout["demo"][-1])
            else:
                sampled_g.append(rollout["demo"][demo_i + j + 1])
        sampled_g = np.stack(sampled_g)

        for i in range(len(rollout["ac"])):
            idx = self._idx = (self._idx + 1) % self._size
            self.sum_tree[idx] = self._max_priority ** self._alpha
            self.min_tree[idx] = self._max_priority ** self._alpha

            self._current_size += 1
            for k in self._keys:
                self._add_value(k, rollout[k][i], idx)
            self._add_value("ob_next", rollout["ob"][i + 1], idx)

            self._add_value("g", sampled_g, idx)

            demo_i = rollout["demo_i"][i + 1]
            sampled_g = []
            for j in range(self._meta_window):
                if demo_i + j + 1 >= len(rollout["demo"]):
                    sampled_g.append(rollout["demo"][-1])
                else:
                    sampled_g.append(rollout["demo"][demo_i + j + 1])
            sampled_g = np.stack(sampled_g)
            self._add_value("g_next", sampled_g, idx)

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, self._current_size - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        assert self._beta > 0
        start = time()

        episode_batch = self._buffers

        # if batch_size > self._current_size:
        #    idxs = np.random.randint(0, self._current_size, batch_size)
        # else:
        idxs = self._sample_proportional(batch_size)

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [episode_batch[key][idx] for idx in idxs]

        # get max weight
        transitions["weight"] = []
        transitions["indexes"] = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self._current_size) ** (-self._beta)
        for i in idxs:
            # calculate weights
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * self._current_size) ** (-self._beta)
            transitions["weight"].append(weight / max_weight)
            transitions["indexes"].append(i)

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)
        return new_transitions

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        self._beta = min(self._beta + 0.001, 1.0)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self._current_size

            self.sum_tree[idx] = priority ** self._alpha
            self.min_tree[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers["ac"])
