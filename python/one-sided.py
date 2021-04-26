from mpi4py import MPI
import numpy as np

W = MPI.COMM_WORLD

rank = W.rank

def separate():
    W.barrier()
    if rank == 0:
        print("", flush = True)
    W.barrier()

if rank == 0:
    data = np.arange(10, dtype = 'i')
    win = MPI.Win.Create(data, comm = W)
else:
    data = np.empty(10, dtype = 'i')
    win = MPI.Win.Create(data, comm = W)

if rank != 0:
    win.Get(data, 0)

print("{}: data: {}".format(rank, data))
