import gzip
import os
import pickle

from torch.utils.data import Dataset


class ILDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(
        self,
        path: str,
        load_mode: str = "demos",
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        """
        load_mode = ['buffer', 'demos']
        """
        self.train = train  # training set or test set
        # TODO: split dataset into train/val

        self._obs = []
        self._acs = []
        self._rews = []
        if load_mode == "demos":
            self._load_demo_files(path)
        elif load_mode == "buffer":
            self._load_replay_buffer(path)

    def _load_demo_files(self, demo_folder, num_demos=float("inf")):
        demos = []
        i = 0
        for d in os.scandir(demo_folder):
            if d.is_file() and d.path.endswith("pkl"):
                demos.append(d.path)
                i += 1
            if i > num_demos:
                break
        # now load the picked numpy arrays
        for file_path in demos:
            with open(file_path, "rb") as f:
                demo = pickle.load(f)

                # add observations
                for ob in demo["obs"]:
                    self._obs.append(ob)
                self._obs.pop()

                # add actions
                for ac in demo["actions"]:
                    self._acs.append(ac)

                # add rewards
                if "rewards" in demo:
                    for rew in demo["rewards"]:
                        self._rews.append(rew)

                assert len(demo["obs"]) == len(demo["actions"]) + 1

    def _load_replay_buffer(self, replay_path: str) -> None:
        with gzip.open(replay_path, "rb") as f:
            replay_buffers = pickle.load(f)
            rb = replay_buffers["replay"]
            self._obs = rb["ob"]
            self._acs = rb["ac"]
            self._rews = rb["rew"]
            assert len(self._obs) == len(self._acs)

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict:{'ob':ob, 'ac':ac} where target is index of the target class.
        """
        ob, ac = self._obs[index], self._acs[index]

        if len(self._rews) > index:
            rew = self._rews[index]
            return {"ob": ob, "ac": ac, "rew": rew}
        else:
            return {"ob": ob, "ac": ac}

    def __len__(self) -> int:
        return len(self._acs)
