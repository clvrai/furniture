import numpy as np
from mpi4py import MPI


def _mpi_average(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    buf /= MPI.COMM_WORLD.Get_size()
    return buf


# Average across the cpu's data
def mpi_average(x):
    if isinstance(x, dict):
        keys = sorted(x.keys())
        return {k: _mpi_average(np.array(x[k])) for k in keys}
    else:
        return _mpi_average(np.array(x))


def _mpi_sum(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    return buf


# Sum over the cpu's data
def mpi_sum(x):
    if isinstance(x, dict):
        keys = sorted(x.keys())
        return {k: _mpi_sum(np.array(x[k])) for k in keys}
    else:
        return _mpi_sum(np.array(x))

