from collections import defaultdict

import numpy as np


class Info(object):
    def __init__(self, info=None):
        if info is None:
            info = defaultdict(list)
        self._info = info.copy()

    def add(self, info):
        if isinstance(info, Info):
            for k, v in info._info.items():
                self._info[k].extend(v)
        elif isinstance(info, dict):
            for k, v in info.items():
                if isinstance(v, list):
                    self._info[k].extend(v)
                else:
                    self._info[k].append(v)
        else:
            raise ValueError("info should be dict or Info (%s)" % info)

    def clear(self):
        self._info = defaultdict(list)

    def get_dict(self, reduction="mean", only_scalar=False):
        ret = {}
        for k, v in self._info.items():
            if np.isscalar(v):
                ret[k] = v
            # elif hassattr(v[0], "shape") and len(v[0].shape) == 1:
            elif isinstance(v[0], (int, float, bool, np.float32, np.int64)):
                if "_mean" in k or reduction == "mean":
                    ret[k] = np.mean(v)
                elif reduction == "sum":
                    ret[k] = np.sum(v)
            elif not only_scalar:
                ret[k] = v
        self.clear()
        return ret

    def __get_item__(self, key):
        return self._info[key]

    def __set_item__(self, key, value):
        self._info[key] = value

    def items(self):
        return self._info.items()
