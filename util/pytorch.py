import os
import io
from glob import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import PIL.Image
from mpi4py import MPI


# Note! This is l2 square, not l2
def l2(a, b):
    return torch.pow(torch.abs(a - b), 2).sum(dim=1)


# required when we load optimizer from a checkpoint
def optimizer_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_ckpt_path(base_dir, ckpt_num):
    if ckpt_num is None:
        return get_recent_ckpt_path(base_dir)
    files = glob(os.path.join(base_dir, "*.pt"))
    for f in files:
        if 'ckpt_%08d.pt' % ckpt_num in f:
            return f, ckpt_num
    raise Exception("Did not find ckpt_%s.pt" % ckpt_num)


def get_recent_ckpt_path(base_dir):
    files = glob(os.path.join(base_dir, "*.pt"))
    files.sort()
    if len(files) == 0:
        return None, None
    max_step = max([f.rsplit('_', 1)[-1].split('.')[0] for f in files])
    paths = [f for f in files if max_step in f]
    if len(paths) == 1:
        return paths[0], int(max_step)
    else:
        raise Exception("Multiple most recent ckpts %s" % paths)


def image_grid(image, n=4):
    return vutils.make_grid(image[:n], nrow=n).cpu().detach().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slice_tensor(input, indices):
    ret = {}
    for k, v in input.items():
        ret[k] = v[indices]
    return ret


def average_gradients(model):
    size = float(dist.get_world_size())
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= size


def ensure_shared_grads(model, shared_model):
    """for A3C"""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def compute_gradient_norm(model):
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += (p.grad.data ** 2).sum().item()
    return grad_norm


def compute_weight_norm(model):
    weight_norm = 0
    for p in model.parameters():
        if p.data is not None:
            weight_norm += (p.data ** 2).sum().item()
    return weight_norm


def compute_weight_sum(model):
    weight_sum = 0
    for p in model.parameters():
        if p.data is not None:
            weight_sum += p.data.abs().sum().item()
    return weight_sum


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)


# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape


# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = np.prod(params_shape[key_name])
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param


# sync gradients across the different cores
def sync_grads(network):
    flat_grads, grads_shape = _get_flat_grads(network)
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_grads(network, grads_shape, global_grads)


def _set_flat_grads(network, grads_shape, flat_grads):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, value in network.named_parameters():
        len_grads = np.prod(grads_shape[key_name])
        copy_grads = flat_grads[pointer:pointer + len_grads].reshape(grads_shape[key_name])
        copy_grads = torch.tensor(copy_grads).to(device)
        # copy the grads
        value.grad.data.copy_(copy_grads.data)
        pointer += len_grads


def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        try:
            grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        except:
            print('Cannot get grad of tensor {}'.format(key_name))
            import pdb; pdb.set_trace()
        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape


def fig2tensor(draw_func):
    def decorate(*args, **kwargs):
        tmp = io.BytesIO()
        fig = draw_func(*args, **kwargs)
        fig.savefig(tmp, dpi=88)
        tmp.seek(0)
        fig.clf()
        return TF.to_tensor(PIL.Image.open(tmp))

    return decorate


def tensor2np(t):
    if isinstance(t, torch.Tensor):
        return t.clone().detach().cpu().numpy()
    else:
        return t


def tensor2img(tensor):
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor.squeeze(0)
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    import cv2
    cv2.imwrite('tensor.png', img)


def obs2tensor(obs, device):
    if isinstance(obs, list):
        obs = list2dict(obs)

    return OrderedDict([
        (k, torch.tensor(np.stack(v), dtype=torch.float32).to(device)) for k, v in obs.items()
    ])


# transfer a numpy array into a tensor
def to_tensor(x, device):
    if isinstance(x, dict):
        return OrderedDict([
            (k, torch.tensor(v, dtype=torch.float32).to(device))
            for k, v in x.items()])
    if isinstance(x, list):
        return [torch.tensor(v, dtype=torch.float32).to(device)
                for v in x]
    return torch.tensor(x, dtype=torch.float32).to(device)


def list2dict(rollout):
    ret = OrderedDict()
    for k in rollout[0].keys():
        ret[k] = []
    for transition in rollout:
        for k, v in transition.items():
            ret[k].append(v)
    return ret


# From softlearning repo
def flatten(unflattened, parent_key='', separator='/'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return OrderedDict(items)


# From softlearning repo
def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result

