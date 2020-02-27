""" Launch RL training and evaluation. """

import sys
import signal
import os
import json

import numpy as np
import torch
from six.moves import shlex_quote
from mpi4py import MPI

from config import argparser
from il.trainer import Trainer
from util.logger import logger


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def run(config):
    """
    Runs Trainer.
    """
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
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        trainer.evaluate()
        logger.info("Finish evaluating")

if __name__ == '__main__':
    args, unparsed = argparser()
    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        run(args)