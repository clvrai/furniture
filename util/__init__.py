
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

