from pyquaternion import Quaternion

class Qpos:
    def __init__(self, x: float, y: float, z: float, quat: Quaternion):
        self.x = x
        self.y = y
        self.z = z
        self.quat = quat

    def __str__(self):
        return '(' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.z) + \
                '), (' + str(self.quat) + ')'


def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        return [int(num) for num in value.split(",")]

def str2set(value):
    if not value:
        return value
    else:
        return set(value.split(","))


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def parse_demo_file_name(file_path):
    # file_path = 'path/to/agent_furniturename_XXXX.pkl'
    parts = file_path.split("/")[-1].split(".")[0].split("_")
    agent_name = parts[0]
    agent_name = agent_name[0].upper() + agent_name[1:]
    furniture_name = "_".join(parts[1:-1])
    return agent_name, furniture_name


def clamp(num, low, high):
    return max(low, min(num, high))
