""" Base code for RL training. Collects rollouts and updates policy networks. """

import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle
import h5py

import torch
import torch.optim as optim
import torch.nn as nn

import wandb
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange

from il.BCDataset import BCDataset
from il.BCModel import *
from il.rollouts import RolloutRunner
from util.logger import logger
from util.pytorch import get_ckpt_path, count_parameters
from util.mpi import mpi_sum
from env import make_env


def get_agent_by_name(config, ob_space, ac_space):
    """
    Returns SAC agent or PPO agent.
    """
    if config.algo == 'BC':
        from il.bc_agent import BCAgent
        return BCAgent(config, obs_space, ac_space)


class Trainer(object):
    """
    Trainer class for SAC and PPO in PyTorch.
    """

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = make_env(config.env, config)
        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        print('***', ac_space)

        # build up networks
        assert config.furniture_name and config.agent_name
        self._agent = get_agent_by_name(config, ob_space, ac_space)
        # build rollout runner
        self._runner = RolloutRunner(
            config, self._env, self._agent
        )

        # # setup log
        # if self._is_chef and self._config.is_train:
        #     exclude = ['device']
        #     if not self._config.wandb:
        #         os.environ['WANDB_MODE'] = 'dryrun'

        #     # user or team name
        #     entity = 'clvr'
        #     # project name
        #     project = 'furniture'

        #     #assert entity != 'clvr', "Please change 'entity' with your wandb id" \
        #     #    "or disable wandb by setting os.environ['WANDB_MODE'] = 'dryrun'"

        #     wandb.init(
        #         resume=config.run_name,
        #         project=project,
        #         config={k: v for k, v in config.__dict__.items() if k not in exclude},
        #         dir=config.log_dir,
        #         entity=entity,
        #         notes=config.notes
        #     )


def train(config):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Parameters
    params = {'batch_size': 8,
              'shuffle': True,
              'num_workers': 8}
    max_epochs = 100

    # Generators
    training_set = BCDataset(self._config.agent_name, self._config.furniture_name)
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    # print(type(train_loader))
    print(len(training_set))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config.bc_lr, momentum=0.9)

    for epoch in range(500):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self._agent.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 10 == 9:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / len(training_set)))
        running_loss = 0.0
    print('Finished Training')

def _evaluate(self, step=None, record=False, idx=None):
    """
    Runs one rollout if in eval mode (@idx is not None).
    Runs num_record_samples rollouts if in train mode (@idx is None).

    Args:
        step: the number of environment steps.
        record: whether to record video or not.
    """
    for i in range(self._config.num_record_samples):
        rollout, info, frames = \
            self._runner.run_episode(is_train=False, record=record)

        if record:
            ep_rew = info['rew']
            ep_success = 's' if info['episode_success'] else 'f'
            fname = '{}_step_{:011d}_{}_r_{}_{}.mp4'.format(
                self._env.name, step, idx if idx is not None else i,
                ep_rew, ep_success)
            video_path = self._save_video(fname, frames)
            # info['video'] = wandb.Video(video_path, fps=15, format='mp4')

        if idx is not None:
            break

    logger.info('rollout: %s', {k: v for k, v in info.items() if not 'qpos' in k})
    self._save_success_qpos(info)
    return rollout, info

    def _save_success_qpos(self, info):
        """ Saves the final qpos of successful trajectory. """
        if self._config.save_qpos and info['episode_success']:
            path = os.path.join(self._config.record_dir, 'qpos.p')
            with h5py.File(path, 'a') as f:
                key_id = len(f.keys())
                num_qpos = len(info['saved_qpos'])
                for qpos_to_save in info['saved_qpos']:
                    f['{}'.format(key_id)] = qpos_to_save
                    key_id += 1
        if self._config.save_success_qpos and info['episode_success']:
            path = os.path.join(self._config.record_dir, 'success_qpos.p')
            with h5py.File(path, 'a') as f:
                key_id = len(f.keys())
                num_qpos = len(info['saved_qpos'])
                for qpos_to_save in info['saved_qpos'][int(num_qpos / 2):]:
                    f['{}'.format(key_id)] = qpos_to_save
                    key_id += 1

    def _save_video(self, fname, frames, fps=15.):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self._config.record_dir, fname)

        def f(t):
            frame_length = len(frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames)/fps+2)

        video.write_videofile(path, fps, verbose=False)
        logger.warn("[*] Video saved: {}".format(path))
        return path


# Loop over epochs


if __name__ == '__main__':
    args, unparsed = argparser()
    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        train(args)

