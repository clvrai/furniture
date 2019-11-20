from collections import OrderedDict
import argparse
import os

import torch

from util.pytorch import get_ckpt_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--env', type=str, default='two-jaco')
    return parser.parse_args()


def switch_key_ant(s):
    s = s.replace('_1', '_2')
    return s


def switch_key_jaco(s):
    s = s.replace('right', 'LEFT')
    s = s.replace('left', 'RIGHT')
    s = s.replace('cube1', 'CUBE2')
    s = s.replace('cube2', 'CUBE1')

    s = s.replace('LEFT', 'left')
    s = s.replace('RIGHT', 'right')
    s = s.replace('CUBE1', 'cube1')
    s = s.replace('CUBE2', 'cube2')

    return s


switch_key = lambda x: switch_key_jaco(x)


def rebuild_ordered_dict(odict):
    return OrderedDict([(switch_key(k), v) for k, v in odict.items()])


def main():
    configs = parse_args()

    in_ckpt_path, _ = get_ckpt_path(configs.in_dir, ckpt_num=configs.ckpt_num)
    out_ckpt_path = in_ckpt_path.replace(configs.in_dir, configs.out_dir)
    os.makedirs(configs.out_dir, exist_ok=True)

    global switch_key
    if configs.env == 'two-jaco':
        switch_key = lambda x: switch_key_jaco(x)
    elif configs.env == 'ant':
        switch_key = lambda x: switch_key_ant(x)
    else:
        raise ValueError('env is not defined')

    ckpt = torch.load(in_ckpt_path)
    ckpt['agent']['actor_state_dict'][0][0] = rebuild_ordered_dict(ckpt['agent']['actor_state_dict'][0][0])
    ckpt['agent']['ob_norm_state_dict'] = rebuild_ordered_dict(ckpt['agent']['ob_norm_state_dict'])
    torch.save(ckpt, out_ckpt_path)


if __name__ == '__main__':
    main()
