import os
from time import time
from collections import defaultdict
from functools import reduce
import gzip
import pickle

from tqdm import tqdm, trange
import torch
import wandb
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import matplotlib.patches as patches
import h5py

from her import get_agent_by_name
from her.meta_agent import MetaAgent
from her.rollouts import RolloutRunner
from her.normalizer import Normalizer
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters
from util.mpi import mpi_sum
from environment import get_env_by_name


class Trainer(object):
    def __init__(self, config):
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = get_env_by_name(config.env)(config)
        ob_space = self._env.observation_space
        goal_space = self._env.goal_space
        ac_space = self._env.dof

        # build up obs normalizers
        if config.ob_norm:
            self._ob_norm = Normalizer(
                ob_space, default_clip_range=config.clip_range, clip_obs=config.clip_obs
            )
            self._g_norm = Normalizer(
                goal_space,
                default_clip_range=config.clip_range,
                clip_obs=config.clip_obs,
            )
        else:
            self._ob_norm = self._g_norm = None

        # load goal estimator
        if config.goal_type == "tcn":
            from types import SimpleNamespace
            from tcn.tcn_estimator import TCNEstimator

            args = SimpleNamespace(
                embedding_dim=config.tcn_dim,
                normalize_embedding=True,
                normalize_radius=20.0,
                pretrained=True,
                checkpoint=config.goal_estimator_path,
                feature_extractor="inception",
            )
            self._goal_estimator = TCNEstimator(args)
        elif config.goal_type == "detector3d":
            from tracking.mujoco_3d_tracker import BasicTrackingEstimator

            self._goal_estimator = BasicTrackingEstimator.from_checkpoint(
                config.goal_estimator_path
            )
        elif config.goal_type == "detector_box":
            from tracking.bb_tracker import BoundingBoxTracker

            self._goal_estimator = BoundingBoxTracker.get_mujoco_bb_tracker(0.8)
        else:
            self._goal_estimator = None

        # build up networks
        self._agent = get_agent_by_name(config.algo)(
            config,
            self._env,
            ob_space,
            goal_space,
            ac_space,
            self._ob_norm,
            self._g_norm,
        )

        self._meta_agent = MetaAgent(
            config, self._env, ob_space, goal_space, self._ob_norm, self._g_norm
        )

        # build rollout runner and replay buffer
        self._runner = RolloutRunner(
            config,
            self._env,
            self._meta_agent,
            self._agent,
            self._goal_estimator,
            self._ob_norm,
            self._g_norm,
        )

        if self._is_chef:
            # debugging
            exclude = ["device", "initial_joint", "state_min", "state_max"]
            if config.debug:
                os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_API_KEY"] = config.wandb_api_key
            wandb_args = dict(
                project="corl_rl",
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity="clvr",
            )
            if config.resume_on_same_name:
                wandb_args["resume"] = config.run_name
            else:
                wandb_args["name"] = config.run_name
            wandb.init(**wandb_args)

    def _save_ckpt(self, ckpt_num, update_iter):
        ckpt_path = os.path.join(self._config.log_dir, "ckpt_%08d.pt" % ckpt_num)
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["meta_agent"] = self._meta_agent.state_dict()
        state_dict["agent"] = self._agent.state_dict()
        if self._config.ob_norm:
            state_dict.update(
                {
                    "ob_norm_state_dict": self._ob_norm.state_dict(),
                    "g_norm_state_dict": self._g_norm.state_dict(),
                }
            )
        torch.save(state_dict, ckpt_path)
        logger.info("Save checkpoint: %s", ckpt_path)

        replay_path = os.path.join(self._config.log_dir, "replay_%08d.pkl" % ckpt_num)
        with gzip.open(replay_path, "wb") as f:
            replay_buffers = {
                "meta_replay": self._meta_agent.replay_buffer(),
                "replay": self._agent.replay_buffer(),
            }
            pickle.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_num=None):
        ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path)
            self._meta_agent.load_state_dict(ckpt["meta_agent"])
            self._agent.load_state_dict(ckpt["agent"])
            if self._config.ob_norm:
                self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
                self._g_norm.load_state_dict(ckpt["g_norm_state_dict"])

            replay_path = os.path.join(
                self._config.log_dir, "replay_%08d.pkl" % ckpt_num
            )
            logger.warn("Load replay_buffer %s", replay_path)
            with gzip.open(replay_path, "rb") as f:
                replay_buffers = pickle.load(f)
                self._meta_agent.load_replay_buffer(replay_buffers["meta_replay"]),
                self._agent.load_replay_buffer(replay_buffers["replay"])

            return ckpt["step"], ckpt["update_iter"]
        else:
            logger.warn("Randomly initialize models")
            return 0, 0

    def _save_rollouts(self, step, meta_rollout, ep_rollout, is_train=True):
        tag = "train" if is_train else "test"
        rollout_path = os.path.join(
            self._config.rollout_dir, tag + ("_rollouts_%08d.pkl" % step)
        )
        with gzip.open(rollout_path, "wb") as f:
            rollouts = {"meta_rollout": meta_rollout, "ep_rollout": ep_rollout}
            pickle.dump(rollouts, f)

    def _log_train(
        self, step, meta_rollout, ep_rollout, meta_train_info, train_info, ep_info
    ):
        if self._env.name == "robot_push":
            logger.info("plotting")
            self._plot_rollout(step, meta_rollout, ep_rollout)
            self._save_rollouts(step, meta_rollout, ep_rollout)

        for k, v in meta_train_info.items():
            wandb.log({"train_meta/%s" % k: v}, step=step)

        for k, v in train_info.items():
            wandb.log({"train_her/%s" % k: v}, step=step)

        for k, v in ep_info.items():
            wandb.log({"train_ep/%s" % k: np.mean(v)}, step=step)

    def _log_test(self, step, meta_rollout, ep_rollout, ep_info):
        # visualize demonstration and observation for robot environment
        if self._env.name == "robot_push":
            logger.info("plotting")
            self._plot_rollout(step, meta_rollout, ep_rollout, is_train=False)
            self._save_rollouts(step, meta_rollout, ep_rollout, is_train=False)

        for k, v in ep_info.items():
            wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)

    def _plot_rollout(self, step, meta_rollout, rollout, is_train=True):
        if self._env.name == "pick_and_place":
            return
        if self._env.name == "furniture":
            return

        # M x 2, M = batchsize, N = rollout size
        meta_idxs = meta_rollout["demo_i"]
        goals = meta_rollout["demo"][:, :2]
        goals = np.asarray(goals)
        trajectories = np.stack([ob["object_ob"][:2] for ob in rollout["ob"]])
        assert trajectories.shape[-1] == 2, trajectories.shape
        assert goals.shape[-1] == 2, goals.shape

        f, ax = plt.subplots()
        if self._env.name == "robot_push":
            plt.xlim(0.45, 0.9)
            plt.ylim(-0.1, 0.1)
        elif self._env.name == "push":
            plt.xlim(0.40, 0.75)
            plt.ylim(-0.12, 0.12)

        # draw goals and meta policy choices
        ax.scatter(goals[:, 0], goals[:, 1], marker=".", color="blue", zorder=2)
        ax.scatter(
            goals[:, 0][meta_idxs],
            goals[:, 1][meta_idxs],
            s=100,
            marker="o",
            facecolor="none",
            edgecolor="black",
            zorder=3,
        )

        (line,) = ax.plot(trajectories[:, 0], trajectories[:, 1], marker=".")
        for i, a in enumerate(trajectories[:-1]):
            b = trajectories[i + 1]
            an = ax.annotate(
                "",
                xy=b[:2],
                xytext=a[:2],
                arrowprops=dict(arrowstyle="simple", color=line.get_color()),
                zorder=1,
            )

        if self._env.name == "robot_push":
            bound_rect = patches.Rectangle(
                (0.671 - 0.01, -0.075 - 0.01),
                0.046 + 0.02,
                0.055 + 0.02,
                0,
                linewidth=1,
                edgecolor="r",
                facecolor="r",
                alpha=0.2,
                hatch="/",
            )
            rect = patches.Rectangle(
                (0.671, -0.075),
                0.046,
                0.055,
                0,
                linewidth=1,
                edgecolor="r",
                facecolor="r",
                alpha=0.8,
            )
        elif self._env.name == "push":
            bound_rect = patches.Rectangle(
                (0.61 - 0.02, 0 - 0.02),
                0.04,
                0.04,
                0,
                linewidth=1,
                edgecolor="r",
                facecolor="r",
                alpha=0.2,
                hatch="/",
            )
            rect = patches.Rectangle(
                (0.61 - 0.02, 0.0 - 0.02),
                0.04,
                0.04,
                0,
                linewidth=1,
                edgecolor="r",
                facecolor="r",
                alpha=0.8,
            )

        ax.add_patch(bound_rect)
        ax.add_patch(rect)
        ax.set_aspect("equal")

        tag = "train" if is_train else "test"
        wandb.log({tag + "_ep/traj": [wandb.Image(plt)]}, step=step)
        fpath = os.path.join(self._config.plot_dir, "{}_{}.png".format(tag, step))
        f.savefig(fpath)
        plt.close("all")

    def train(self):
        config = self._config
        meta_agent = self._meta_agent
        num_batches = config.num_batches

        # load checkpoint
        step, update_iter = self._load_ckpt()

        # sync the networks across the cpus
        self._agent.sync_networks()
        self._meta_agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(
                initial=step, total=config.max_global_step, desc=config.run_name
            )
            ep_info = defaultdict(list)

        st_time = time()
        st_step = step
        while step < config.max_global_step:
            rollout, meta_rollout, info = self._runner.run_episode()
            logger.info("rollout: %s", info)
            self._agent.store_episode(rollout)
            self._meta_agent.store_episode(meta_rollout)
            if step < config.max_ob_norm_step:
                self._update_normalizer(rollout)
            step_per_batch = mpi_sum(info["len"])

            logger.info("Update meta networks %d", update_iter)
            meta_train_info = self._meta_agent.train()
            logger.info("Update meta networks done")

            logger.info("Update networks %d", update_iter)
            train_info = self._agent.train()
            logger.info("Update networks done")

            step += step_per_batch
            update_iter += 1

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
                    self._log_train(
                        step,
                        meta_rollout,
                        rollout,
                        meta_train_info,
                        train_info,
                        ep_info,
                    )
                    ep_info = defaultdict(list)

                if update_iter % config.evaluate_interval == 0:
                    logger.info("Evaluate at %d", update_iter)
                    rollout, meta_rollout, info = self._evaluate(
                        step=step, record=config.record
                    )
                    self._log_test(step, meta_rollout, rollout, info)

                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def _update_normalizer(self, rollout):
        if self._config.ob_norm:
            self._ob_norm.update(rollout["ob"])
            self._g_norm.update(rollout["g"] + rollout["ag"])
            self._ob_norm.recompute_stats()
            self._g_norm.recompute_stats()

    def _evaluate(self, step=None, record=False, idx=None):
        rollout, meta_rollout, info = self._runner.run_episode(
            is_train=False, record=record, step=step, idx=idx
        )
        logger.info("eval rollout: %s", info)
        return rollout, meta_rollout, info

    def evaluate(self):
        step, update_iter = self._load_ckpt(ckpt_num=self._config.ckpt_num)

        logger.info(
            "Run %d evaluations at step=%d, update_iter=%d",
            self._config.num_eval,
            step,
            update_iter,
        )
        info_history = defaultdict(list)
        for i in trange(self._config.num_eval):
            logger.warn("Evalute run %d", i + 1)
            rollout, meta_rollout, info = self._evaluate(
                step=step, record=self._config.record, idx=i
            )
            self._plot_rollout(i, meta_rollout, rollout, is_train=False)
            self._save_rollouts(i, meta_rollout, rollout, is_train=False)
            for k, v in info.items():
                info_history[k].append(v)

        keys = [
            "hrl_success",
            "env_success",
            "covered_frames",
            "demo_len",
            "coverage",
            "fail_low",
            "fail_meta",
        ]
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k in [
                "hrl_success",
                "env_success",
                "covered_frames",
                "demo_len",
                "fail_low",
                "fail_meta",
            ]:
                hf.create_dataset(k, data=info_history[k])

            coverage = []
            for x, y in zip(info_history["covered_frames"], info_history["demo_len"]):
                coverage.append(int(x) / int(y))
            hf.create_dataset("coverage", data=coverage)

            result = "{:.02f} $\\pm$ {:.02f}".format(
                np.mean(info_history["env_success"]),
                np.std(info_history["env_success"]),
            )
            logger.warn(result)
