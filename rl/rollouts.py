"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

from collections import defaultdict

import numpy as np
import cv2

from util.logger import logger


class Rollout(object):
    """
    Rollout storing an episode.
    """

    def __init__(self):
        """ Initialize buffer. """
        self._history = defaultdict(list)

    def add(self, data):
        """ Add a transition @data to rollout buffer. """
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        """ Returns rollout buffer and clears buffer. """
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ac'] = self._history['ac']
        batch['ac_before_activation'] = self._history['ac_before_activation']
        batch['done'] = self._history['done']
        batch['rew'] = self._history['rew']
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    """
    Run rollout given environment and policy.
    """

    def __init__(self, config, env, pi):
        """
        Args:
            config: configurations for the environment.
            env: environment.
            pi: policy.
        """

        self._config = config
        self._env = env
        self._pi = pi

    def run_episode(self, max_step=10000, is_train=True, record=False):
        """
        Runs one episode and returns the rollout.

        Args:
            max_step: maximum number of steps of the rollout.
            is_train: whether rollout is for training or evaluation.
            record: record videos of rollout if True.
        """
        config = self._config
        device = config.device
        env = self._env
        pi = self._pi
        gail = config.algo == "gail"

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        if gail:
            ep_rew_gail = 0
        ob = self._env.reset()

        self._record_frames = []
        if record: self._store_frame()

        # run rollout
        while not done and ep_len < max_step:
            # sample action from policy
            ac, ac_before_activation = pi.act(ob, is_train=is_train)

            rollout.add({'ob': ob, 'ac': ac, 'ac_before_activation': ac_before_activation})

            if gail:
                reward_gail = pi.predict_reward(ob, ac)

            # take a step
            ob, reward, done, info = env.step(ac)

            rollout.add({'done': done,
                         'rew': reward_gail if gail else reward})
            acs.append(ac)
            ep_len += 1
            ep_rew += reward
            if gail:
                ep_rew_gail += reward_gail

            for key, value in info.items():
                reward_info[key].append(value)
            if record:
                frame_info = info.copy()
                if gail:
                    frame_info.update({"ep_rew_gail": ep_rew_gail,
                                       "rew_gail": reward_gail})
                self._store_frame(frame_info)

        # last frame
        rollout.add({'ob': ob})

        # compute average/sum of information
        ep_info = {'len': ep_len, 'rew': ep_rew}
        if gail:
            ep_info["rew_gail"] = ep_rew_gail
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool, np.float32)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)

        return rollout.get(), ep_info, self._record_frames

    def _store_frame(self, info={}):
        """ Renders a frame and stores in @self._record_frames. """
        color = (200, 200, 200)

        # render video frame
        frame = self._env.render('rgb_array')[0] * 255.0
        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

        if self._config.record_caption:
            # add caption to video frame
            text = "{:4} {}".format(self._env._episode_length,
                                    self._env._episode_reward)
            font_size = 0.4
            thickness = 1
            offset = 12
            x, y = 5, fheight + 10
            cv2.putText(frame, text,
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 0), thickness, cv2.LINE_AA)
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = '{}: '.format(k)
                (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, thickness)

                cv2.putText(frame, key_text,
                            (x, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (66, 133, 244), thickness, cv2.LINE_AA)

                cv2.putText(frame, str(v),
                            (x + key_width, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (255, 255, 255), thickness, cv2.LINE_AA)

        self._record_frames.append(frame)
