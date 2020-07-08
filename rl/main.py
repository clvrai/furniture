""" Launch RL training and evaluation. """

import json
import os
import signal
import sys

import numpy as np
import torch
from mpi4py import MPI
from six.moves import shlex_quote

from config import argparser
from rl.reset_trainer import ResetTrainer
from rl.trainer import Trainer
from util.logger import logger
from util.mpi import mpi_sync

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def run(config):
    """
    Runs Trainer.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.num_workers = MPI.COMM_WORLD.Get_size()
    set_log_path(config)

    config.seed = config.seed + rank
    # config.port = config.port + rank * 2 # training env + evaluation env

    if config.is_chef:
        logger.warn("Run a base worker.")
        make_log_files(config)
    else:
        logger.warn("Run worker %d and disable logger.", config.rank)
        import logging

        logger.setLevel(logging.CRITICAL)

    # syncronize all processes
    mpi_sync()

    def shutdown(signal, frame):
        logger.warn("Received signal %s: exiting", signal)
        sys.exit(128 + signal)

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
    if config.algo == "bc":
        trainer = Trainer(config)
    elif config.algo in ["ppo", "ddpg"]:
        trainer = Trainer(config)
    else:
        trainer = ResetTrainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        if config.record_demo:
            trainer.record_demos()
            logger.info("Finish generating demos")
        else:
            trainer.evaluate()
            logger.info("Finish evaluating")


def set_log_path(config):
    """
    Sets paths to log directories.
    """
    method = "il" if config.algo in ["bc", "gail"] else "rl"
    config.run_name = "{}.{}.{}.{}.{}".format(
        method, config.env, config.algo, config.prefix, config.seed
    )
    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    config.record_dir = os.path.join(config.log_dir, "video")
    config.plot_dir = os.path.join(config.log_dir, "plot")


def make_log_files(config):
    """
    Sets up log directories and saves git diff and command line.
    """
    logger.info("Create log directory: %s", config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    logger.info("Create video directory: %s", config.record_dir)
    os.makedirs(config.record_dir, exist_ok=True)

    logger.info("Create plot directory: %s", config.plot_dir)
    os.makedirs(config.plot_dir, exist_ok=True)

    if config.is_train:
        # log git diff
        cmds = [
            "echo `git rev-parse HEAD` >> {}/git.txt".format(config.log_dir),
            "git diff >> {}/git.txt".format(config.log_dir),
            "echo 'python -m rl {}' >> {}/cmd.sh".format(
                " ".join([shlex_quote(arg) for arg in sys.argv[1:]]), config.log_dir
            ),
        ]
        os.system("\n".join(cmds))

        # log config
        param_path = os.path.join(config.log_dir, "params.json")
        logger.info("Store parameters in %s", param_path)
        with open(param_path, "w") as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    args, unparsed = argparser()
    if len(unparsed):
        logger.error("Unparsed argument is detected:\n%s", unparsed)
    else:
        run(args)
