import sys
import signal
import os
import json

import numpy as np
import torch
from six.moves import shlex_quote
from mpi4py import MPI

from util.logger import logger


np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)


def get_trainer(config):
    if config.method == 'rl':
        from her.trainer import Trainer
        return Trainer(config)
    else:
        raise Exception('The method is not avaiable %s' % config.method)


def get_agent_by_name(algo):
    if algo == 'sac':
        from her.sac_agent import SACAgent
        return SACAgent
    elif algo == 'ddpg':
        from her.ddpg_agent import DDPGAgent
        return DDPGAgent


def run(config):
    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size()

    if config.is_chef:
        logger.warn('Run a base worker.')
        make_log_files(config)
    else:
        logger.warn('Run worker %d and disable logger.', config.rank)
        import logging
        logger.setLevel(logging.CRITICAL)

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    # build a trainer
    trainer = get_trainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        trainer.evaluate()
        logger.info("Finish evaluating")


def make_log_files(config):
    config.run_name = 'her.{}.{}.{}'.format(config.env, config.prefix, config.seed)
    config.log_dir = os.path.join(config.log_dir, config.run_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    config.plot_dir = os.path.join(config.log_dir, 'plots')
    os.makedirs(config.plot_dir, exist_ok=True)

    config.rollout_dir = os.path.join(config.log_dir, 'rollouts')
    os.makedirs(config.rollout_dir, exist_ok=True)

    if config.is_train:
        cmds = [
            "echo `git rev-parse HEAD` >> {}/git.txt".format(config.log_dir),
            "git diff >> {}/git.txt".format(config.log_dir),
            "echo 'python -m her.{} {}' >> {}/cmd.sh".format(
                config.env,
                ' '.join([shlex_quote(arg) for arg in sys.argv[1:]]),
                config.log_dir),
        ]
        os.system("\n".join(cmds))

    param_path = os.path.join(config.log_dir, 'params.json')
    logger.info('Store parameters in %s', param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def main(config, unparsed):
    if len(unparsed):
        logger.error('Unparsed config is detected:\n%s', unparsed)
        return

    run(config)

