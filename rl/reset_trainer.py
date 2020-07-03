"""
Base code for RL and IL training.
Collects rollouts and updates policy networks.
"""

import copy
import gzip
import os
import pickle
from collections import defaultdict
from typing import Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from tqdm import tqdm

from env import make_env
from rl import get_agent_by_name
from rl.aot_agent import AoTAgent
from rl.policies import get_actor_critic_by_name
from rl.rollouts import Rollout
from rl.sac_agent import SACAgent
from rl.trainer import Trainer
from util.logger import logger
from util.mpi import mpi_sum
from util.pytorch import get_ckpt_path


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
        ac_space = self._env.action_space
        rev_space = self._env.reversible_space

        f_actor, f_critic = get_actor_critic_by_name(config.policy, config.algo)
        self._agent: SACAgent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, f_actor, f_critic
        )

        self._aot_agent: AoTAgent = None
        rew = None
        if config.use_aot:

            self._aot_agent = AoTAgent(
                config, rev_space, self._agent._buffer, self._env.get_reverse
            )
            rew = self._aot_agent.rew

        actor, critic = get_actor_critic_by_name(config.policy, config.algo)
        self._reset_agent: SACAgent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, actor, critic, True, rew
        )

        if config.reset_kl_penalty:
            self._reset_agent.set_forward_actor(self._agent._actor)
        if config.safe_forward:
            self._agent.set_reset_critic(
                self._reset_agent._critic1_target, self._reset_agent._critic2_target
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
        if cfg.reset_init_ckpt_path:
            self._load_reset_policy(cfg.reset_init_ckpt_path)
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
            ep_len = ep_rew = safe_act = 0
            env.begin_forward()
            while not done:  # env return done if time limit is reached or task done
                ac, ac_before_activation = self._agent.act(ob, is_train=True)
                if self._config.safe_forward and not self._agent.is_safe_action(ob, ac):
                    ac, ac_before_activation = self._agent.safe_act(ob, is_train=True)
                    safe_act += 1
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
            ep_info.update({"len": ep_len, "rew": ep_rew, "safe_act": safe_act})
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
                    ob, env_reward, reset_done, info = env.step(ac)
                    # env_reward, reset_rew_info = env.reset_reward(prev_ob, ac, ob)
                    reward = env_reward
                    if cfg.use_aot:
                        intr_reward = self._aot_agent.rew(prev_ob, ob)[0]
                        info["intr_reward"] = intr_reward
                        info["env_reward"] = env_reward
                        reward += intr_reward
                    info["reward"] = reward
                    reset_success = env.reset_success()
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
                # train reset agent
                r_rollout = r_rollout.get()
                self._reset_agent.store_episode(r_rollout)
                if cfg.use_aot and cfg.aot_success_buffer:
                    self._aot_agent.store_episode(r_rollout, success=reset_success)
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
            if self._is_chef:
                if not cfg.status_quo_baseline:
                    self._pbar.update(step_per_batch)
                if self._update_iter % cfg.log_interval == 0:
                    r_ep_info.update({"total_reset": total_reset})
                    self._log_train(self._step, r_train_info, r_ep_info, "reset")

            self._update_iter += 1
            self._evaluate(record=True)
            self._save_ckpt(self._step, self._update_iter)

    @torch.no_grad()
    def _evaluate(self, record=False) -> Tuple[dict, dict, dict, dict]:
        """
        Runs cfg.num_eval rollouts of reset rl.
        """
        if not (
            self._is_chef and self._update_iter % self._config.evaluate_interval == 0
        ):
            return
        cfg = self._config
        ep_info_history = defaultdict(list)
        r_ep_info_history = defaultdict(list)

        def gather_ep_history(history, ep):
            for key, value in ep.items():
                if isinstance(value, wandb.Video):
                    history[key] = value
                else:
                    history[key].append(value)

        eval_rollouts = []
        for i in range(cfg.num_eval):
            rec = record and i == 0
            rollout, ep_info, r_rollout, r_ep_info = self._evaluate_rollout(record=rec)
            eval_rollouts.append([rollout, ep_info, r_rollout, r_ep_info])
            gather_ep_history(ep_info_history, ep_info)
            gather_ep_history(r_ep_info_history, r_ep_info)

        # summarize ep infos
        ep_info = self._reduce_info(ep_info_history, "mean")
        self._log_test(self._step, ep_info, "forward")
        r_ep_info = self._reduce_info(r_ep_info_history, "mean")
        self._log_test(self._step, r_ep_info, "reset")
        if not cfg.status_quo_baseline and cfg.use_aot:
            self._visualize_aot(eval_rollouts[:2])

    def _evaluate_rollout(self, record=False) -> Tuple[dict, dict, dict, dict]:
        """
        Runs one rollout if in eval mode (@idx is not None) with seed as @idx.
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            record: whether to record video or not.
            log: whether to log results
        Returns:
            rollout, rollout info, reset rollout, reset rollout info
        """
        cfg = self._config
        # 1. Run forward policy
        ep_info = defaultdict(list)
        rollout = Rollout()
        done = False
        ep_len = ep_rew = safe_act = 0
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
            if self._config.safe_forward and not self._agent.is_safe_action(ob, ac):
                ac, ac_before_activation = self._agent.safe_act(ob, is_train=False)
                safe_act += 1
            rollout.add(
                {"ob": ob, "ac": ac, "ac_before_activation": ac_before_activation}
            )
            ob, reward, done, info = env.step(ac)
            done = done or ep_len >= cfg.max_episode_steps
            rollout.add({"done": done, "rew": reward, "safe_act": safe_act})
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

        if cfg.status_quo_baseline:
            return rollout, ep_info, {}, {}

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
            ob, env_reward, reset_done, info = env.step(ac)
            # env_reward, reset_rew_info = env.reset_reward(prev_ob, ac, ob)
            reward = env_reward
            if cfg.use_aot:
                intr_reward = self._aot_agent.rew(prev_ob, ob)[0]
                info["intr_reward"] = intr_reward
                info["env_reward"] = env_reward
                reward += intr_reward
            info["reward"] = reward
            reset_success = env.reset_success()
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
        return rollout, ep_info, r_rollout, r_ep_info

    def _load_reset_policy(self, ckpt_path):
        """
        Used to only load the reset policy
        """
        logger.warn("Load reset checkpoint %s", ckpt_path)
        ckpt = torch.load(ckpt_path)
        self._reset_agent.load_policy(ckpt["reset_agent"])

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
                    if self._aot_agent and self._config.aot_success_buffer:
                        self._aot_agent.load_replay_buffer(
                            replay_buffers["aot_success"]
                        )
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
                if self._aot_agent and self._config.aot_success_buffer:
                    replay_buffers["aot_success"] = self._aot_agent.replay_buffer()
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

    def _visualize_aot(self, rollouts: list):
        """
        Visualize AoT and Reset Policy
        Get AoT outputs for each trajectory
        Plot onto 2D space using PCA or TSNE.
        Color each point by value of AoT. Scale size of each point by variance.
        Args:
             rollouts: [forward, forward_info, reset, reset_info]
        """
        pca = PCA(2)
        X = []
        aots = []
        trajectories = []
        ensemble = self._config.aot_ensemble is not None
        if ensemble:
            trajectories_sizes = []
        for i in range(len(rollouts)):
            f, f_info, r, r_info = rollouts[i]

            # plot the centroid of peg instead of top position
            def visualize_ob(ob):
                if self._config.env == "PegInsertion":
                    obs = self._env.get_reverse(ob)
                    return obs[:3] + 0.5 * (obs[3:] - obs[:3])
                elif self._config.env == "FurnitureSawyerPushEnv":
                    obs = self._env.get_reverse(ob)
                    return obs

            fob = [visualize_ob(x) for x in f["ob"]]
            rob = [visualize_ob(x) for x in r["ob"]]
            X.extend(fob)
            X.extend(rob)
            fout = self._aot_agent.act(f["ob"], is_train=False, return_info=ensemble)
            rout = self._aot_agent.act(r["ob"], is_train=False, return_info=ensemble)
            f_aot = fout
            r_aot = rout
            if ensemble:
                f_aot, f_info = fout
                r_aot, r_info = rout
                # calculate sizes for each trajectory
                f_var = f_info["var"].cpu().numpy().flatten()  # (N, 1)
                r_var = r_info["var"].cpu().numpy().flatten()  # (N, 1)
                # rescale the variances to min_s and max_s
                min_s = 2
                max_s = 80
                f_min_var, r_min_var = np.min(f_var), np.min(r_var)
                f_max_var, r_max_var = np.max(f_var), np.max(r_var)
                f_var = min_s + (f_var - f_min_var) * (max_s - min_s) / (
                    f_max_var - f_min_var
                )
                r_var = min_s + (r_var - r_min_var) * (max_s - min_s) / (
                    r_max_var - r_min_var
                )
                trajectories_sizes.extend([f_var, r_var])
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
        ax.set_title(f"Update iter: {self._update_iter}, Step: {self._step}")
        ax.set_facecolor((0, 0, 0, 0.1))
        for j, traj in enumerate(trajectories):
            traj = pca.transform(traj)
            aot = aots[j]
            s = trajectories_sizes[j] if ensemble else 2
            # draw dots with arrows for traj
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                marker=".",
                s=s,
                c=aot,
                cmap=cmap,
                norm=norm,
                zorder=2,
            )
            # annotate start and end points of trajectory
            dir = "f" if j % 2 == 0 else "r"
            if dir == "f":
                ax.annotate(f"s_f", traj[0, :], fontsize="xx-small", color="b")
            ax.annotate(f"e_{dir}", traj[-1, :], fontsize="xx-small", color="b")
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

    def _reduce_info(self, ep_info, reduction="sum"):
        for key, value in ep_info.items():
            if isinstance(value, wandb.Video):
                continue
            elif isinstance(value[0], (int, float, bool, np.float32)):
                if "_mean" in key or reduction == "mean":
                    ep_info[key] = np.mean(value)
                elif reduction == "sum":
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
    config.plot_dir = ""

    trainer = ResetTrainer(config)
    trainer._update_iter = 0
    trainer._step = 0
    trainer._evaluate(False)


if __name__ == "__main__":
    test_aot_visualization()
