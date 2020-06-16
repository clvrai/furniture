"""
Base code for RL and IL training.
Collects rollouts and updates policy networks.
"""

import copy
import gzip
import os
import pickle
from collections import defaultdict
from time import time
from typing import Tuple

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import wandb
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

from env import make_env
from rl.policies import get_actor_critic_by_name
from rl.rollouts import Rollout, RolloutRunner
from util.logger import logger
from util.mpi import mpi_sum
from util.pytorch import get_ckpt_path


def get_agent_by_name(algo):
    """
    Returns RL or IL agent.
    """
    if algo == "sac":
        from rl.sac_agent import SACAgent

        return SACAgent
    elif algo == "ppo":
        from rl.ppo_agent import PPOAgent

        return PPOAgent
    elif algo == "ddpg":
        from rl.ddpg_agent import DDPGAgent

        return DDPGAgent
    elif algo == "bc":
        from il.bc_agent import BCAgent

        return BCAgent
    elif algo == "gail":
        from il.gail_agent import GAILAgent

        return GAILAgent


class Trainer(object):
    """
    Trainer class for SAC, PPO, DDPG, BC, and GAIL in PyTorch.
    """

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef
        self._is_rl = config.algo in ["ppo", "sac"]

        # create environment
        self._env_eval = (
            make_env(config.env, copy.copy(config)) if self._is_chef else None
        )

        config.unity = False  # disable Unity for training
        self._env = make_env(config.env, config)
        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        logger.info("Action space: " + str(ac_space))

        # get actor and critic networks
        actor, critic = get_actor_critic_by_name(config.policy, config.algo)

        # build up networks
        self._agent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, actor, critic
        )

        # build rollout runner
        self._runner = RolloutRunner(config, self._env, self._env_eval, self._agent)

        # setup log
        if self._is_chef and self._config.is_train:
            exclude = ["device"]
            if not self._config.wandb:
                os.environ["WANDB_MODE"] = "dryrun"

            # user or team name
            entity = "clvr"
            # project name
            project = "reverse"

            # assert entity != 'clvr', "Please change 'entity' with your wandb id" \
            #    "or disable wandb by setting os.environ['WANDB_MODE'] = 'dryrun'"

            wandb.init(
                resume=config.run_name,
                project=project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=entity,
                notes=config.notes,
            )

    def _save_ckpt(self, ckpt_num, update_iter):
        """
        Save checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
        """
        ckpt_path = os.path.join(self._config.log_dir, "ckpt_%08d.pt" % ckpt_num)
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["agent"] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

        if self._config.algo in ["sac", "ddpg"]:
            replay_path = os.path.join(
                self._config.log_dir, "replay_%08d.pkl" % ckpt_num
            )
            with gzip.open(replay_path, "wb") as f:
                replay_buffers = {"replay": self._agent.replay_buffer()}
                pickle.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_path=None, ckpt_num=None):
        """
        Loads checkpoint with path @ckpt_path or index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        if ckpt_path is None:
            ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)
        else:
            ckpt_num = int(ckpt_path.rsplit("_", 1)[-1].split(".")[0])

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path)
            self._agent.load_state_dict(ckpt["agent"])

            if self._config.is_train and self._config.algo in ["sac", "ddpg"]:
                replay_path = os.path.join(
                    self._config.log_dir, "replay_%08d.pkl" % ckpt_num
                )
                logger.warn("Load replay_buffer %s", replay_path)
                with gzip.open(replay_path, "rb") as f:
                    replay_buffers = pickle.load(f)
                    self._agent.load_replay_buffer(replay_buffers["replay"])
            return ckpt["step"], ckpt["update_iter"]
        else:
            logger.warn("Randomly initialize models")
            return 0, 0

    def _log_train(self, step, train_info, ep_info):
        """
        Logs training and episode information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({"train_rl/%s" % k: v}, step=step)
            else:
                wandb.log({"train_rl/%s" % k: [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({"train_ep/%s" % k: np.mean(v)}, step=step)
            wandb.log({"train_ep_max/%s" % k: np.max(v)}, step=step)

    def _log_test(self, step, ep_info):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                if isinstance(v, wandb.Video):
                    wandb.log({"test_ep/%s" % k: v}, step=step)
                else:
                    wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)

    def train(self):
        """ Trains an agent. """
        if self._is_rl:
            self._train_rl()
        else:
            self._train_il()

    def _train_il(self):
        """ Trains an IL agent. """
        config = self._config

        # load checkpoint
        step, update_iter = self._load_ckpt()
        if config.init_ckpt_path:
            self._load_ckpt(ckpt_path=config.init_ckpt_path)

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(
                initial=update_iter, total=config.max_epoch, desc=config.run_name
            )

        # decide how many episodes or how long rollout to collect
        if self._config.algo == "gail":
            runner = self._runner.run(every_steps=self._config.rollout_length)
        elif self._config.algo == "bc":
            runner = None

        st_time = time()
        st_step = step
        while update_iter < config.max_epoch:
            # collect rollouts
            if runner:
                rollout, info = next(runner)
                self._agent.store_episode(rollout)
                step_per_batch = mpi_sum(len(rollout["ac"]))
            else:
                step_per_batch = mpi_sum(1)

            # train an agent
            logger.info("Update networks %d", update_iter)
            train_info = self._agent.train()

            logger.info("Update networks done")

            if self._config.algo != "bc" and step < config.max_ob_norm_step:
                self._update_normalizer(rollout)

            step += step_per_batch
            update_iter += 1

            # log training and episode information or evaluate
            if self._is_chef:
                pbar.update(1)

                if update_iter % config.log_interval == 0:
                    train_info.update(
                        {
                            "sec": (time() - st_time) / config.log_interval,
                            "steps_per_sec": (step - st_step) / (time() - st_time),
                            "update_iter": update_iter,
                        }
                    )
                    st_time = time()
                    st_step = step
                    self._log_train(step, train_info, {})

                if update_iter % config.evaluate_interval == 0:
                    logger.info("Evaluate at %d", update_iter)
                    # rollout, info = self._evaluate(step=step, record=config.record)
                    # self._log_test(step, info)
                    rollout, info = self._bc_evaluate(step)
                    self._log_test(step, info)

                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def _train_rl(self):
        """ Trains an RL agent. """
        config = self._config

        # load checkpoint
        step, update_iter = self._load_ckpt()
        if config.init_ckpt_path:
            self._load_ckpt(ckpt_path=config.init_ckpt_path)

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(
                initial=step, total=config.max_global_step, desc=config.run_name
            )
            ep_info = defaultdict(list)

        # decide how many episodes or how long rollout to collect
        if self._config.algo == "ppo":
            runner = self._runner.run(every_steps=self._config.rollout_length)
        elif self._config.algo == "sac":
            runner = self._runner.run(every_steps=1)

        st_time = time()
        st_step = step
        while step < config.max_global_step:
            # collect rollouts
            rollout, info = next(runner)
            self._agent.store_episode(rollout)

            step_per_batch = mpi_sum(len(rollout["ac"]))

            # train an agent
            logger.info("Update networks %d", update_iter)
            train_info = self._agent.train()

            logger.info("Update networks done")

            if step < config.max_ob_norm_step:
                self._update_normalizer(rollout)

            step += step_per_batch
            update_iter += 1

            # log training and episode information or evaluate
            if self._is_chef:
                pbar.update(step_per_batch)

                if update_iter % config.log_interval == 0:
                    for k, v in info.items():
                        if isinstance(v, list):
                            ep_info[k].extend(v)
                        else:
                            ep_info[k].append(v)
                    train_info.update(
                        {
                            "sec": (time() - st_time) / config.log_interval,
                            "steps_per_sec": (step - st_step) / (time() - st_time),
                            "update_iter": update_iter,
                        }
                    )
                    st_time = time()
                    st_step = step
                    self._log_train(step, train_info, ep_info)
                    ep_info = defaultdict(list)

                if update_iter % config.evaluate_interval == 1:
                    logger.info("Evaluate at %d", update_iter)
                    rollout, info = self._evaluate(step=step, record=config.record)
                    self._log_test(step, info)

                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def _update_normalizer(self, rollout):
        """ Updates normalizer with @rollout. """
        if self._config.ob_norm:
            self._agent.update_normalizer(rollout["ob"])

    def _bc_evaluate(self, step, mode="sample", num_eval=10):
        """
        Evaluates the BC policy over a subset or entire test dataset
        """
        if mode == "sample":
            logger.info("Run %d evaluations at step=%d", num_eval, step)
            info_history = defaultdict(list)
            for i in tqdm(range(num_eval), desc="evaluating..."):
                record = i == 0
                rollout, info, frames = self._runner.run_episode(
                    record=record, is_train=False
                )
                if record:
                    ep_rew = info["rew"]
                    ep_success = "s" if info["episode_success"] else "f"
                    fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                        self._env.name, step, i, ep_rew, ep_success,
                    )
                    video_path = self._save_video(fname, frames)
                    info_history["video"] = wandb.Video(
                        video_path, fps=15, format="mp4"
                    )
                for k, v in info.items():
                    info_history[k].append(v)

            return rollout, info_history
        elif mode == "all":
            num_eval = len(self._env.seed_test)
            logger.info("Run %d evaluations at step=%d", num_eval, step)
            info_history = defaultdict(list)
            for i in tqdm(range(num_eval), desc="evaluating..."):
                rollout, info, frames = self._runner.run_episode(is_train=False, seed=i)
                for k, v in info.items():
                    info_history[k].append(v)

            return rollout, info_history

    def _evaluate(self, step=None, record=False, idx=None, record_demo=False):
        """
        Runs one rollout if in eval mode (@idx is not None) with seed as @idx.
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
            record: whether to record video or not.
        """
        logger.info(
            "Run %d evaluations at step=%d", self._config.num_record_samples, step
        )

        for i in range(self._config.num_record_samples):
            rollout, info, frames = self._runner.run_episode(
                is_train=False, record=record, record_demo=record_demo, seed=idx
            )

            if record:
                ep_rew = info["rew"]
                ep_success = "s" if info["episode_success"] else "f"
                fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    self._env.name,
                    step,
                    idx if idx is not None else i,
                    ep_rew,
                    ep_success,
                )
                video_path = self._save_video(fname, frames)
                info["video"] = wandb.Video(video_path, fps=15, format="mp4")

            if idx is not None:
                break

        logger.info("rollout: %s", {k: v for k, v in info.items() if "qpos" not in k})
        return rollout, info

    def evaluate(self):
        """ Evaluates an agent stored in chekpoint with @self._config.ckpt_num. """
        step, update_iter = self._load_ckpt(ckpt_num=self._config.ckpt_num)

        logger.info(
            "Run %d evaluations at step=%d, update_iter=%d",
            self._config.num_eval,
            step,
            update_iter,
        )
        info_history = defaultdict(list)
        rollouts = []
        for i in trange(self._config.num_eval):
            logger.warn("Evalute run %d", i + 1)
            rollout, info = self._evaluate(step=step, record=self._config.record, idx=i)
            for k, v in info.items():
                info_history[k].append(v)
            if self._config.save_rollout:
                rollouts.append(rollout)

        keys = ["episode_success", "reward_goal_dist"]
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k in keys:
                hf.create_dataset(k, data=info_history[k])

            result = "{:.02f} $\\pm$ {:.02f}".format(
                np.mean(info_history["episode_success"]),
                np.std(info_history["episode_success"]),
            )
            logger.warn(result)

        if self._config.save_rollout:
            os.makedirs("saved_rollouts", exist_ok=True)
            with open("saved_rollouts/{}.p".format(self._config.run_name), "wb") as f:
                pickle.dump(rollouts, f)

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

    def record_demos(self) -> None:
        """
        Record num_eval demos. Pass in the idx so we can record over all demos
        """
        step, update_iter = self._load_ckpt()
        if self._config.init_ckpt_path:
            self._load_ckpt(ckpt_path=self._config.init_ckpt_path)

        logger.info(
            "Run %d demonstration collection at step=%d, update_iter=%d",
            self._config.num_eval,
            step,
            update_iter,
        )
        for i in trange(self._config.num_eval):
            logger.warn("Evalute run %d", i + 1)
            rollout, info = self._evaluate(
                step=step, record=self._config.record, record_demo=True
            )


class ResetTrainer(Trainer):
    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef
        self._is_rl = config.algo in ["ppo", "sac"]

        self._build_envs()
        self._build_agents()
        self._setup_log()

    def _build_envs(self):
        config = self._config
        # create environment
        self._env_eval = (
            make_env(config.env, copy.copy(config)) if self._is_chef else None
        )
        config.unity = False  # disable Unity for training
        self._env = make_env(config.env, config)

    def _build_agents(self):
        config = self._config
        ob_space = self._env.observation_space
        goal_space = self._env.goal_space
        ac_space = self._env.action_space

        actor, critic = get_actor_critic_by_name(config.policy, config.algo)
        self._agent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, actor, critic
        )

        self._aot_agent = None
        rew = None
        if config.use_aot:
            from rl.aot_agent import AoTAgent

            self._aot_agent = AoTAgent(
                config, goal_space, self._agent._buffer, self._env.get_goal
            )
            rew = self._aot_agent.rew

        actor, critic = get_actor_critic_by_name(config.policy, config.algo)
        self._reset_agent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, actor, critic, True, rew
        )

    def _setup_log(self):
        # setup log
        config = self._config
        if self._is_chef and self._config.is_train:
            exclude = ["device"]
            if not self._config.wandb:
                os.environ["WANDB_MODE"] = "dryrun"

            # user or team name
            entity = "clvr"
            # project name
            project = "resetrl"

            # assert entity != 'clvr', "Please change 'entity' with your wandb id" \
            #    "or disable wandb by setting os.environ['WANDB_MODE'] = 'dryrun'"

            wandb.init(
                resume=config.run_name,
                project=project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=entity,
                notes=config.notes,
            )

    def train(self):
        """
        Training loop for reset RL.
        1. Train forward policy
        2. Train AoT model on forward policy
        3. Train Reset policy
        4. Reset env if needed
        Repeat
        """
        cfg = self._config
        total_reset = reset_fail = 0
        env = self._env

        # load checkpoint
        self._step, self._update_iter = self._load_ckpt()
        if cfg.init_ckpt_path:
            self._load_ckpt(ckpt_path=cfg.init_ckpt_path)
        # sync the networks across the cpus
        self._agent.sync_networks()
        self._reset_agent.sync_networks()
        if self._aot_agent is not None:
            self._aot_agent.sync_networks()

        if self._is_chef:
            self._pbar = tqdm(
                initial=self._step, total=cfg.max_global_step, desc=cfg.run_name
            )

        ob = env.reset(is_train=True)
        while self._step < cfg.max_global_step:
            # 1. Run and train forward policy
            rollout = Rollout()
            ep_info = defaultdict(list)
            done = False
            ep_len = ep_rew = 0
            env.begin_forward()
            while not done:  # env return done if time limit is reached or task done
                ac, ac_before_activation = self._agent.act(ob, is_train=True)
                rollout.add(
                    {"ob": ob, "ac": ac, "ac_before_activation": ac_before_activation}
                )
                ob, reward, done, info = env.step(ac)
                done = done or ep_len >= cfg.max_episode_steps
                rollout.add({"done": done, "rew": reward})
                ep_len += 1
                ep_rew += reward
                for key, value in info.items():
                    ep_info[key].append(value)
            # last frame
            rollout.add({"ob": ob})
            # compute average/sum of information
            ep_info = self._reduce_info(ep_info)
            ep_info.update({"len": ep_len, "rew": ep_rew})
            # train forward agent
            rollout = rollout.get()
            self._agent.store_episode(rollout)
            train_info = self._agent.train()
            self._update_normalizer(rollout, self._agent)
            step_per_batch = mpi_sum(len(rollout["ac"]))
            self._step += step_per_batch
            if self._is_chef:
                self._pbar.update(step_per_batch)
                if self._update_iter % cfg.log_interval == 0:
                    train_info.update({"update_iter": self._update_iter})
                    self._log_train(self._step, train_info, ep_info, "forward")
            if cfg.status_quo_baseline:
                total_reset += 1
                ob = env.reset(is_train=True)
                r_train_info = {}
                r_ep_info = {}
            else:
                # 2. Update AoT Classifier
                if self._aot_agent is not None:
                    arrow_train_info = self._aot_agent.train()
                    if self._is_chef and self._update_iter % cfg.log_interval == 0:
                        self._log_train(self._step, arrow_train_info, {}, "aot")

                # 3. Run and train reset policy
                r_rollout = Rollout()
                r_ep_info = defaultdict(list)
                reset_done = reset_success = False
                ep_len = ep_rew = 0
                env.begin_reset()
                while not reset_done:
                    ac, ac_before_activation = self._reset_agent.act(ob, is_train=True)
                    prev_ob = ob
                    r_rollout.add(
                        {
                            "ob": ob,
                            "ac": ac,
                            "ac_before_activation": ac_before_activation,
                        }
                    )
                    ob, _, _, info = env.step(ac)
                    env_reward, reset_rew_info = env.reset_reward(prev_ob, ac, ob)
                    reward = env_reward
                    if cfg.use_aot:
                        intr_reward = self._aot_agent.rew(prev_ob, ob)[0]
                        info["intr_reward"] = intr_reward
                        info["env_reward"] = env_reward
                        reward += intr_reward
                    info["reward"] = reward
                    info.update(reset_rew_info)
                    for k in [
                        "dist_to_goal",
                        "control_rew",
                        "peg_to_goal_rew",
                        "success_rew",
                    ]:
                        del info[k]

                    reset_success = env.reset_done()
                    info["episode_success"] = int(reset_success)
                    reset_done = reset_success or ep_len >= cfg.max_reset_episode_steps
                    # don't add rew to rollout because it gets computed online
                    if cfg.use_aot:
                        r_rollout.add({"done": reset_done, "env_rew": env_reward})
                    else:
                        r_rollout.add({"done": reset_done, "rew": reward})
                    ep_len += 1
                    ep_rew += reward
                    for key, value in info.items():
                        r_ep_info[key].append(value)
                # last frame
                r_rollout.add({"ob": ob})
                # compute average/sum of information
                r_ep_info = self._reduce_info(r_ep_info)
                r_ep_info.update({"len": ep_len, "rew": ep_rew})
                # train forward agent
                r_rollout = r_rollout.get()
                self._reset_agent.store_episode(r_rollout)
                r_train_info = self._reset_agent.train()
                self._update_normalizer(r_rollout, self._reset_agent)
                step_per_batch = mpi_sum(len(r_rollout["ac"]))
                self._step += step_per_batch

                # 4. Hard Reset if necessary
                if not reset_success:
                    reset_fail += 1
                    if reset_fail % cfg.max_failed_reset == 0:
                        reset_fail = 0
                        total_reset += 1
                        ob = env.reset(is_train=True)
                else:
                    logger.info("successful learned reset")
            if self._is_chef:
                if not cfg.status_quo_baseline:
                    self._pbar.update(step_per_batch)
                if self._update_iter % cfg.log_interval == 0:
                    r_ep_info.update({"total_reset": total_reset})
                    self._log_train(self._step, r_train_info, r_ep_info, "reset")

            self._update_iter += 1
            self._evaluate(record=True)
            self._save_ckpt(self._step, self._update_iter)

    def _evaluate(self, record=False, log=True) -> Tuple[dict, dict, dict, dict]:
        """
        Runs one rollout if in eval mode (@idx is not None) with seed as @idx.
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            record: whether to record video or not.
            log: whether to log results
        Returns:
            rollout, rollout info, reset rollout, reset rollout info
        """
        if not (
            self._is_chef and self._update_iter % self._config.evaluate_interval == 0
        ):
            return
        cfg = self._config
        # 1. Run forward policy
        ep_info = defaultdict(list)
        rollout = Rollout()
        done = False
        ep_len = ep_rew = 0
        env = self._env_eval

        forward_frames = []
        reset_frames = []

        ob = env.reset(is_train=False)
        if record:
            frame = env.render("rgb_array")[0] * 255.0
            forward_frames.append(frame)
        env.begin_forward()
        while not done:  # env return done if time limit is reached or task done
            ac, ac_before_activation = self._agent.act(ob, is_train=False)
            rollout.add(
                {"ob": ob, "ac": ac, "ac_before_activation": ac_before_activation}
            )
            ob, reward, done, info = env.step(ac)
            done = done or ep_len >= cfg.max_episode_steps
            rollout.add({"done": done, "rew": reward})
            ep_len += 1
            ep_rew += reward
            for key, value in info.items():
                ep_info[key].append(value)
            if record:
                frame = env.render("rgb_array")[0] * 255.0
                forward_frames.append(frame)
        rollout.add({"ob": ob})
        rollout = rollout.get()
        ep_info = self._reduce_info(ep_info)
        ep_info.update({"len": ep_len, "rew": ep_rew})
        if record:
            ep_rew = ep_info["reward"]
            ep_success = "s" if ep_info["episode_success"] else "f"
            fname = f"forward_step_{self._step:011d}_r_{ep_rew:.3f}_{ep_success}.mp4"
            video_path = self._save_video(fname, forward_frames)
            ep_info["video"] = wandb.Video(video_path, fps=15, format="mp4")
        if log:
            self._log_test(self._step, ep_info, "forward")

        if cfg.status_quo_baseline:
            return

        # 3. Run and train reset policy
        r_rollout = Rollout()
        r_ep_info = defaultdict(list)
        reset_done = reset_success = False
        ep_len = ep_rew = 0
        if record:
            frame = env.render("rgb_array")[0] * 255.0
            reset_frames.append(frame)
        env.begin_reset()
        while not reset_done:
            ac, ac_before_activation = self._reset_agent.act(ob, is_train=False)
            prev_ob = ob
            r_rollout.add(
                {"ob": ob, "ac": ac, "ac_before_activation": ac_before_activation}
            )
            ob, _, _, info = env.step(ac)
            env_reward, reset_rew_info = env.reset_reward(prev_ob, ac, ob)
            reward = env_reward
            if cfg.use_aot:
                intr_reward = self._aot_agent.rew(prev_ob, ob)[0]
                info["intr_reward"] = intr_reward
                info["env_reward"] = env_reward
                reward += intr_reward
            info["reward"] = reward
            reset_success = env.reset_done()
            info["episode_success"] = int(reset_success)
            reset_done = reset_success or ep_len >= cfg.max_reset_episode_steps
            # don't add rew to rollout because it gets computed online
            if cfg.use_aot:
                r_rollout.add({"done": reset_done, "env_rew": env_reward})
            else:
                r_rollout.add({"done": reset_done, "rew": reward})
            ep_len += 1
            ep_rew += reward
            for key, value in info.items():
                r_ep_info[key].append(value)
            if record:
                frame = env.render("rgb_array")[0] * 255.0
                reset_frames.append(frame)
        r_rollout.add({"ob": ob})
        r_rollout = r_rollout.get()
        r_ep_info = self._reduce_info(r_ep_info)
        r_ep_info.update({"len": ep_len, "rew": ep_rew})
        if record:
            ep_rew = r_ep_info["reward"]
            ep_success = "s" if r_ep_info["episode_success"] else "f"
            fname = f"reset_step_{self._step:011d}_r_{ep_rew:.3f}_{ep_success}.mp4"
            video_path = self._save_video(fname, reset_frames)
            r_ep_info["video"] = wandb.Video(video_path, fps=15, format="mp4")
        if log:
            self._log_test(self._step, r_ep_info, "reset")
            if cfg.use_aot:
                self._visualize_aot()

        return rollout, ep_info, r_rollout, r_ep_info

    def _load_ckpt(self, ckpt_path=None, ckpt_num=None):
        """
        Loads checkpoint with path @ckpt_path or index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        if ckpt_path is None:
            ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)
        else:
            ckpt_num = int(ckpt_path.rsplit("_", 1)[-1].split(".")[0])

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path)
            self._agent.load_state_dict(ckpt["agent"])
            self._reset_agent.load_state_dict(ckpt["reset_agent"])
            if self._config.use_aot:
                self._aot_agent.load_state_dict(ckpt["aot_agent"])

            if self._config.is_train and self._config.algo in ["sac", "ddpg"]:
                replay_path = os.path.join(
                    self._config.log_dir, "replay_%08d.pkl" % ckpt_num
                )
                logger.warn("Load replay_buffer %s", replay_path)
                with gzip.open(replay_path, "rb") as f:
                    replay_buffers = pickle.load(f)
                    self._agent.load_replay_buffer(replay_buffers["replay"])
                    self._reset_agent.load_replay_buffer(replay_buffers["reset_replay"])
            return ckpt["step"], ckpt["update_iter"]
        else:
            logger.warn("Randomly initialize models")
            return 0, 0

    def _save_ckpt(self, ckpt_num, update_iter):
        """
        Save checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
        """
        if not (self._is_chef and update_iter % self._config.ckpt_interval == 0):
            return
        ckpt_path = os.path.join(self._config.log_dir, "ckpt_%08d.pt" % ckpt_num)
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["agent"] = self._agent.state_dict()
        state_dict["reset_agent"] = self._reset_agent.state_dict()
        if self._config.use_aot:
            state_dict["aot_agent"] = self._aot_agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

        if self._config.algo in ["sac", "ddpg"]:
            replay_path = os.path.join(
                self._config.log_dir, "replay_%08d.pkl" % ckpt_num
            )
            with gzip.open(replay_path, "wb") as f:
                replay_buffers = {
                    "replay": self._agent.replay_buffer(),
                    "reset_replay": self._reset_agent.replay_buffer(),
                }
                pickle.dump(replay_buffers, f)

    def _log_test(self, step, ep_info, agent):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                if isinstance(v, wandb.Video):
                    wandb.log({f"{agent}_test_ep/{k}": v}, step=step)
                else:
                    wandb.log({f"{agent}_test_ep/{k}": np.mean(v)}, step=step)

    def _log_train(self, step, train_info, ep_info, agent):
        """
        Logs training and episode information to wandb.
        Args:
            step: the number of environment steps.
            train_info: agent information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({f"{agent}_train_rl/{k}": v}, step=step)
            else:
                wandb.log({f"{agent}_train_rl/{k}": [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({f"{agent}_train_ep/{k}": v}, step=step)

    def _visualize_aot(self):
        """
        Visualize AoT and Reset Policy
        Sample 5 forward policy trajectories
        Sample 5 reset policy trajectories
        Get AoT outputs for each trajectory
        Plot onto 2D space using PCA or TSNE. Color each point by value of AoT
        """
        pca = PCA(2)
        X = []
        aots = []
        trajectories = []
        # Get 2 forward and reset trajectories
        for i in range(2):
            f, f_info, r, r_info = self._evaluate(record=False, log=False)
            fob = [self._env.get_goal(x)[:3] for x in f["ob"]]
            rob = [self._env.get_goal(x)[:3] for x in r["ob"]]
            X.extend(fob)
            X.extend(rob)
            f_aot = self._aot_agent.act(f["ob"], is_train=False)
            r_aot = self._aot_agent.act(r["ob"], is_train=False)
            f_aot = f_aot.cpu().detach().numpy().flatten()
            r_aot = r_aot.cpu().detach().numpy().flatten()
            aot_min = min(f_aot.min(), r_aot.min())
            aot_max = max(f_aot.max(), r_aot.max())
            if i == 0:
                amin = aot_min
                amax = aot_max
            if aot_min < amin:
                amin = aot_min
            if aot_max > amax:
                amax = aot_max

            aots.extend([f_aot, r_aot])
            trajectories.extend([fob, rob])
        levels = MaxNLocator(nbins=15).tick_values(amin, amax)
        cmap = plt.get_cmap("hot")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        pca.fit(X)

        f, ax = plt.subplots()
        ax.set_facecolor((0, 0, 0, 0.1))
        for j, traj in enumerate(trajectories):
            traj = pca.transform(traj)
            aot = aots[j]
            # draw dots with arrows for traj
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                marker=".",
                s=2,
                c=aot,
                cmap=cmap,
                norm=norm,
                zorder=2,
            )
            # annotate start and end points of trajectory
            dir = "f" if j % 2 == 0 else "r"
            ax.annotate(f"s_{dir}", traj[0, :], fontsize="xx-small")
            ax.annotate(f"e_{dir}", traj[-1, :], fontsize="xx-small")
        f.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical"
        )

        fpath = os.path.join(self._config.plot_dir, f"aot_{self._step:011d}.png")
        wandb.log({"reset_test_ep/eval_plot": [wandb.Image(plt)]}, step=self._step)
        f.savefig(fpath, dpi=300)
        plt.close("all")

    def _update_normalizer(self, rollout, agent):
        if self._step < self._config.max_ob_norm_step and self._config.ob_norm:
            agent.update_normalizer(rollout["ob"])

    def _reduce_info(self, ep_info):
        for key, value in ep_info.items():
            if isinstance(value[0], (int, float, bool, np.float32)):
                if "_mean" in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)
        return ep_info


def test_aot_visualization():
    from config import create_parser

    parser = create_parser("PegInsertionEnv")
    config, _ = parser.parse_known_args()
    config.use_aot = True
    config.is_chef = True
    config.device = torch.device("cpu")
    config.is_train = False
    config.wandb = False

    trainer = ResetTrainer(config)
    trainer._update_iter = 0
    trainer._visualize_aot()


if __name__ == "__main__":
    test_aot_visualization()
