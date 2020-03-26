def str2bool(v):
    return v.lower() == 'true'


def str2intlist(value):
    if not value:
        return value
    else:
       return [int(num) for num in value.split(',')]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(',')]


def parse_demo_file_name(file_path):
    # file_path = 'path/to/agent_furniturename_XXXX.pkl'
    parts = file_path.split('/')[-1].split('.')[0].split('_')
    agent_name = parts[0]
    agent_name = agent_name[0].upper() + agent_name[1:]
    furniture_name = '_'.join(parts[1:-1])
    return agent_name, furniture_name

def clamp(num, low, high):
    return max(low, min(num, high))