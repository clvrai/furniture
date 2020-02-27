import torch
import torch.optim as optim
import torch.nn as nn

from il.BCDataset import BCDataset
from il.BCModel import *
from config import argparser

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
    training_set = BCDataset('Cursor', 'swivel_chair_0700')
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    # print(type(train_loader))

    print(len(training_set))

    obs_space, action_space = training_set.getObsActs()
    net = getBCModel(obs_space, action_space)

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
            outputs = net(inputs)
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
            info['video'] = wandb.Video(video_path, fps=15, format='mp4')

        if idx is not None:
            break

    logger.info('rollout: %s', {k: v for k, v in info.items() if not 'qpos' in k})
    self._save_success_qpos(info)
    return rollout, info



# Loop over epochs


if __name__ == '__main__':
    args, unparsed = argparser()
    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        train(args)
