from collections import defaultdict

import numpy as np
import torch
import cv2
from util.logger import logger
import matplotlib.pyplot as plt


class Rollout(object):
    def __init__(self, visual_ob):
        self._history = defaultdict(list)
        self._visual_ob = visual_ob

    def add(self, data):
        for key, value in data.items():
            if not self._visual_ob and 'ob' in key and 'normal' in value:
                value = {k: v for k, v in value.items() if k != 'normal'}
            self._history[key].append(value)

    def get(self):
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ag'] = self._history['ag']
        batch['g'] = self._history['g']
        batch['ac'] = self._history['ac']
        batch['rew'] = self._history['rew']
        self._history = defaultdict(list)
        return batch


class MetaRollout(Rollout):
    def get(self):
        batch = {}
        batch['demo'] = self._history['demo'][0]
        batch['ob'] = self._history['meta_ob']
        batch['demo_i'] = self._history['meta_ac']
        batch['ac'] = self._history['meta_next_ac']
        batch['rew'] = self._history['meta_rew']
        batch['done'] = self._history['meta_done']
        self._history = defaultdict(list)
        return batch

    def extend(self, data):
        for value in data:
            self.add(value)


class RolloutRunner(object):
    def __init__(self, config, env, meta_pi, pi, g_estimator, ob_norm, g_norm):
        self._config = config
        self._env = env
        self._meta_pi = meta_pi
        self._pi = pi
        self._g_estimator = g_estimator

        if config.ob_norm:
            self._ob_norm = ob_norm.normalize
            self._g_norm = g_norm.normalize
        else:
            self._ob_norm = self._g_norm = lambda x:x

    def _new_game(self, is_train=True, record=False, idx=None):
        if idx is None:
            seed = None
        else:
            seed = self._env.seed_test[idx]
        ob, seed = self._env.new_game(is_train=is_train, record=record, seed=seed)
        demo = self._env.load_demo(seed)
        return seed, ob, demo

    def _get_goal_emb(self, ob):
        if isinstance(ob, list):
            ob = np.stack(ob)
        if np.max(ob) <= 1:
            ob = ob * 255
        return self._g_estimator.infer(ob.astype(np.uint8)).cpu().numpy()

    def run_episode(self, is_train=True, record=False, step=None, idx=None):
        config = self._config
        device = config.device
        env = self._env
        meta_pi = self._meta_pi
        pi = self._pi
        ob_norm = self._ob_norm
        g_norm = self._g_norm

        rollout = Rollout(self._config.visual_ob)
        meta_rollout = MetaRollout(self._config.visual_ob)
        reward_info = defaultdict(list)
        acs = []
        meta_acs = []

        done = False
        ep_len = 0
        hrl_reward = 0
        hrl_success = False
        skipped_frames = 0
        is_possible_goal = None

        while True:
            meta_ac = 0
            max_meta_ac = 0
            covered_frames = 0
            seed, ob, demo = self._new_game(is_train=is_train,
                                            record=record,
                                            idx=idx)

            if self._g_estimator:
                demo['goal'] = self._get_goal_emb(demo['frames'])

            # tracker must restart before rollout
            if self._config.goal_type == "detector_box":
                self._g_estimator.restart()

            self._record_frames = []
            self._record_agent_frames = []
            if record: self._store_frame(ob, None, demo, meta_ac, {})

            ag = self._get_goal_emb(ob['normal']) if self._g_estimator else env.get_goal(ob)
            while meta_ac < len(demo['goal']) - 1 and env.is_success(ag, demo['goal'][meta_ac + 1]):
                meta_ac += 1
                covered_frames += 1
                max_meta_ac = meta_ac
                if record: self._store_frame(ob, None, demo, meta_ac, {})

            if meta_ac == len(demo['goal']) - 1:
                logger.error('The demonstration is covered by the initial frame: %d', seed)
            else:
                break

        meta_rollout.add({'demo': demo['goal']})

        # run rollout
        while not done:
            # meta policy
            meta_obs = []
            meta_ac_prev = meta_ac
            meta_obs.append(ob)
            meta_next_ac = meta_pi.act(ob_norm(ob), g_norm(demo['goal']), meta_ac, is_train=is_train)
            if meta_next_ac == -1:
                import ipdb; ipdb.set_trace()
            meta_ac += meta_next_ac + 1
            meta_acs.append(meta_next_ac)
            g = demo['goal'][meta_ac]
            skipped_frames += meta_next_ac

            # low-level policy
            meta_len = 0
            meta_rew = 0.0
            while not done:
                meta_len += 1
                ep_len += 1
                ac = pi.act(ob_norm(ob), g_norm(g), is_train=is_train)
                meta_obs.append(ob)
                rollout.add({'ob': ob, 'ag': ag, 'g': g, 'ac': ac})

                ob, reward, done, info = env.step(ac)
                ag = self._get_goal_emb(ob['normal']) if self._g_estimator else env.get_goal(ob)
                if np.any(np.isnan(ag)):
                    logger.error("Encounter untrackable frame, end the rollout")
                    done = True

                rollout.add({'done': done, 'rew': reward})
                acs.append(ac)

                for key, value in info.items():
                    reward_info[key].append(value)

                if record: self._store_frame(ob, ac, demo, meta_ac, info)

                if env.is_success(ag, g):
                    covered_frames += 1
                    meta_rew += 1.0
                    max_meta_ac = meta_ac
                    break

            if env.is_success(ag, g):
                if env.name == 'robot_push':
                    logger.info('robot_push achieved demo goal {}: {}'.format(meta_ac, g))
                while meta_ac < len(demo['goal']) - 1 and env.is_success(ag, demo['goal'][meta_ac + 1]):
                    meta_ac += 1
                    covered_frames += 1
                    max_meta_ac = meta_ac
                    g = demo['goal'][meta_ac]
                    if record: self._store_frame(ob, None, demo, meta_ac, {})

                if meta_ac == len(demo['goal']) - 1:
                    meta_rew += self._config.completion_bonus
                    hrl_success = True
                    done = True

            else:
                is_possible_goal = env.is_possible_goal(g)

            for meta_ob in meta_obs:
                meta_rollout.add({'meta_ob': meta_ob, 'meta_ac':  meta_ac_prev,
                                  'meta_rew': meta_rew, 'meta_done': done * 1.0,
                                  'meta_next_ac': meta_next_ac})
            hrl_reward += meta_rew

        # last frame
        rollout.add({'ob': ob, 'ag': ag})
        meta_rollout.add({'meta_ob': ob, 'meta_ac': meta_ac})
        env_success = env.get_env_success(ob, demo['goal_gt'])

        if record:
            self._env.frames = self._record_frames
            self._env.scores = hrl_reward
            fname = '{}_step_{:011d}_({})_{}_{}_{}.{}_{}.mp4'.format(
                self._env.name, step, seed, idx, hrl_reward, max_meta_ac, len(demo['goal']) - 1,
                'success' if hrl_success else 'fail')
            self._env.save_video(fname=fname)

            self._env.frames = self._record_agent_frames
            self._env.scores = hrl_reward
            fname = '{}_step_{:011d}_({})_{}_{}_{}.{}_{}_agent.mp4'.format(
                self._env.name, step, seed, idx, hrl_reward, max_meta_ac, len(demo['goal']) - 1,
                'success' if hrl_success else 'fail')
            self._env.save_video(fname=fname)

        ep_info = {'len': ep_len,
                   'demo_len': len(demo['goal']),
                   'max_meta_ac': max_meta_ac,
                   'mean_ac': np.mean(np.abs(acs)),
                   'mean_meta_ac': np.mean(meta_acs),
                   'hrl_rew': hrl_reward,
                   'hrl_success': hrl_success,
                   'covered_frames': covered_frames,
                   'skipped_frames': skipped_frames,
                   'env_success': env_success}
        if is_possible_goal is not None:
            if is_possible_goal:
                ep_info['fail_low'] = 1.0
            else:
                ep_info['fail_meta'] = 1.0

        for key, value in reward_info.items():
            if isinstance(value[0], (int, float)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)

        return rollout.get(), meta_rollout.get(), ep_info

    def _store_frame(self, ob, ac, demo, meta_ac, info):
        color = (200, 200, 200)

        action_txt = ''
        if ac is not None:
            action_txt = ','.join(['%.03f' % a for a in ac.tolist()])
        text = "{:4} ({}/{}) {}".format(self._env._episode_length,
                                        meta_ac, len(demo['goal']) - 1, action_txt)
        if not self._config.caption: text = ''
        frame = ob['normal']
        if not self._env.name == 'robot_push':
            frame = frame * 255.0
        if ac is not None:
            self._record_agent_frames.append(frame)
        h, w, c = frame.shape
        new_frame = np.zeros([h, 2*w, c])
        new_frame[:, 0:w, :] = frame
        if self._env.name == 'robot_push':
            new_frame[:,w:2*w,:] = 0
        else:
            h_, w_, c_ = demo['frames'][meta_ac].shape
            new_frame[0:h_, w:w+w_, :] = demo['frames'][meta_ac]

        if self._config.screen_width == 128:
            font_size = 0.3
        else:
            font_size = 0.4
        thickness = 1
        x, y = 5, 10
        cv2.putText(new_frame, text,
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, color, thickness, cv2.LINE_AA)
        x, y = 5, 25
        if self._g_estimator:
            goal_dist = np.linalg.norm(demo['goal'][meta_ac] - self._get_goal_emb(ob['normal']))
            text = "goal_dist ({:2f})".format(goal_dist)
            cv2.putText(new_frame, text,
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, color, thickness, cv2.LINE_AA)
        if self._config.goal_type == 'detector_box':
            cx_g, cy_g, w_g, h_g = demo['goal'][meta_ac]
            cx_o, cy_o, w_o, h_o = self._get_goal_emb(ob['normal'])
            x1_g, y1_g, x2_g, y2_g = cx_g - w_g / 2, cy_g - h_g / 2, cx_g + w_g / 2, cy_g + h_g / 2
            x1_o, y1_o, x2_o, y2_o = cx_o - w_o / 2, cy_o - h_o / 2, cx_o + w_o / 2, cy_o + h_o / 2
            cv2.rectangle(new_frame, (int(x1_o), int(y1_o)), (int(x2_o), int(y2_o)), color, 1)
            cv2.rectangle(new_frame, (int(x1_g)+w, int(y1_g)), (int(x2_g)+w, int(y2_g)), color, 1)

        self._record_frames.append(new_frame)

