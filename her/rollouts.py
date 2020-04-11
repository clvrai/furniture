import os
from collections import defaultdict

import cv2
import moviepy.editor as mpy
import numpy as np

from util.logger import logger


class Rollout(object):
    def __init__(self, visual_ob):
        self._history = defaultdict(list)
        self._visual_ob = visual_ob

    def add(self, data):
        for key, value in data.items():
            if not self._visual_ob and "ob" in key and "normal" in value:
                value = {k: v for k, v in value.items() if k != "normal"}
            self._history[key].append(value)

    def get(self):
        batch = {}
        batch["ob"] = self._history["ob"]
        batch["ag"] = self._history["ag"]
        batch["g"] = self._history["g"]
        batch["ac"] = self._history["ac"]
        batch["rew"] = self._history["rew"]
        self._history = defaultdict(list)
        return batch


class MetaRollout(Rollout):
    def get(self):
        batch = {}
        batch["demo"] = self._history["demo"][0]
        batch["ob"] = self._history["meta_ob"]
        batch["demo_i"] = self._history["meta_ac"]
        batch["ac"] = self._history["meta_next_ac"]
        batch["rew"] = self._history["meta_rew"]
        batch["done"] = self._history["meta_done"]
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
            self._ob_norm = self._g_norm = lambda x: x

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

    def run_episode(
        self, is_train=True, record=False, step=None, idx=None, record_demo=False
    ):
        """
        Runs one full metarollout. Returns the metarollout and rollout and info.
        """
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
            seed, ob, demo = self._new_game(is_train=is_train, record=record, idx=idx)

            if self._g_estimator:
                demo["goal"] = self._get_goal_emb(demo["frames"])

            # tracker must restart before rollout
            if self._config.goal_type == "detector_box":
                self._g_estimator.restart()

            self._record_frames = []
            self._record_agent_frames = []
            if record:
                self._store_frame(env, {})

            ag = (
                self._get_goal_emb(ob["normal"])
                if self._g_estimator
                else env.get_goal(ob)
            )
            """
            First skip past all frames in the demonstration
            that are already satisfied. Then begin the meta rollout.
            """
            while meta_ac < len(demo["goal"]) - 1 and env.is_success(
                ag, demo["goal"][meta_ac + 1]
            ):
                meta_ac += 1
                # don't count initial skipping as covered
                # covered_frames += 1
                max_meta_ac = meta_ac
                if record:
                    self._store_frame(env, {})

            if meta_ac == len(demo["goal"]) - 1:
                logger.error(
                    "The demonstration is covered by the initial frame: %d", seed
                )
            else:
                break

        meta_rollout.add({"demo": demo["goal"]})

        # run rollout
        while not done:
            # meta policy
            meta_obs = []
            meta_ac_prev = meta_ac
            meta_obs.append(ob)
            meta_next_ac = meta_pi.act(
                ob_norm(ob), g_norm(demo["goal"]), meta_ac, is_train=is_train
            )
            if meta_next_ac == -1:
                import ipdb

                ipdb.set_trace()
            meta_ac += meta_next_ac + 1
            meta_acs.append(meta_next_ac)
            g = demo["goal"][meta_ac]
            skipped_frames += meta_next_ac

            # low-level policy
            meta_len = 0
            meta_rew = 0.0
            low_level_len = 0
            gcp_out_of_time = False
            while not done:
                meta_len += 1
                ep_len += 1
                low_level_len += 1
                ac = pi.act(ob_norm(ob), g_norm(g), is_train=is_train)
                meta_obs.append(ob)
                rollout.add({"ob": ob, "ag": ag, "g": g, "ac": ac})

                ob, reward, done, info = env.step(ac)
                ag = (
                    self._get_goal_emb(ob["normal"])
                    if self._g_estimator
                    else env.get_goal(ob)
                )
                if np.any(np.isnan(ag)):
                    logger.error("Encounter untrackable frame, end the rollout")
                    done = True

                rollout.add({"done": done, "rew": reward})
                acs.append(ac)

                for key, value in info.items():
                    reward_info[key].append(value)

                if record:
                    self._store_frame(env, info)

                if env.is_success(ag, g):
                    covered_frames += 1
                    meta_rew += 1.0
                    max_meta_ac = meta_ac
                    break

                if low_level_len > self._config.gcp_horizon:
                    logger.debug("gcp out of time")
                    gcp_out_of_time = True
                    break
            goal_success = env.is_success(ag, g)
            if goal_success or gcp_out_of_time:
                """
                Once we have achieved the current subgoal,
                find the next subgoal, taking care to not choose already
                feasible subgoals.
                Alternatively, if subpolicy ran out of time, go to the
                next frame anyways.
                """
                while meta_ac < len(demo["goal"]) - 1 and env.is_success(
                    ag, demo["goal"][meta_ac + 1]
                ):
                    meta_ac += 1
                    if goal_success:
                        covered_frames += 1
                    max_meta_ac = meta_ac
                    g = demo["goal"][meta_ac]
                    if record:
                        self._store_frame(env, {})

                if meta_ac == len(demo["goal"]) - 1:
                    meta_rew += self._config.completion_bonus
                    hrl_success = goal_success
                    done = True

            else:
                is_possible_goal = env.is_possible_goal(g)

            for meta_ob in meta_obs:
                meta_rollout.add(
                    {
                        "meta_ob": meta_ob,
                        "meta_ac": meta_ac_prev,
                        "meta_rew": meta_rew,
                        "meta_done": done * 1.0,
                        "meta_next_ac": meta_next_ac,
                    }
                )
            hrl_reward += meta_rew

        # last frame
        rollout.add({"ob": ob, "ag": ag})
        meta_rollout.add({"meta_ob": ob, "meta_ac": meta_ac})
        env_success = env.get_env_success(ob, demo["goal_gt"])

        ep_info = {
            "len": ep_len,
            "demo_len": len(demo["goal"]),
            "max_meta_ac": max_meta_ac,
            "mean_ac": np.mean(np.abs(acs)),
            "mean_meta_ac": np.mean(meta_acs),
            "hrl_rew": hrl_reward,
            "hrl_success": hrl_success,
            "covered_frames": covered_frames,
            "skipped_frames": skipped_frames,
            "env_success": env_success,
        }
        if record:
            self._env.frames = self._record_frames
            fname = "{}_step_{:011d}_({})_{}_{}_{}.{}_{}.mp4".format(
                self._env.name,
                step,
                seed,
                idx,
                hrl_reward,
                max_meta_ac,
                len(demo["goal"]) - 1,
                "success" if hrl_success else "fail",
            )
            video_path = self._save_video(fname=fname, frames=self._record_frames)
            ep_info["video"] = video_path
        if record_demo:
            self._env.save_demo()

        if is_possible_goal is not None:
            if is_possible_goal:
                ep_info["fail_low"] = 1.0
            else:
                ep_info["fail_meta"] = 1.0

        for key, value in reward_info.items():
            if isinstance(value[0], (int, float)):
                if "_mean" in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)

        return rollout.get(), meta_rollout.get(), ep_info

    def _store_frame(self, env, info={}):
        """
        Renders a frame and stores in @self._record_frames.
        """
        # render video frame
        frame = env.render("rgb_array")[0] * 255.0
        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

        if self._config.record_caption:
            # add caption to video frame
            text = "{:4} {}".format(env._episode_length, env._episode_reward)
            font_size = 0.4
            thickness = 1
            offset = 12
            x, y = 5, fheight + 10
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = "{}: ".format(k)
                (key_width, _), _ = cv2.getTextSize(
                    key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                )

                cv2.putText(
                    frame,
                    key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (66, 133, 244),
                    thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        self._record_frames.append(frame)

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self._config.record_dir, fname)
        logger.warn("[*] Generating video: {}".format(path))

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(path, fps, verbose=False)
        logger.warn("[*] Video saved: {}".format(path))
        return path