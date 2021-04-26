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
    win = MPI.Win.Create(None, comm = W)

separate()

print("{}: data: {}".format(rank, data))

separate()

win.Fence()

if rank != 0:
   win.Get(data, 0)

win.Fence()

print("{}: data: {}".format(rank, data))

separate()

if rank != 0:
  win.Accumulate(data, 0, op = MPI.SUM)

win.Fence()

print("{}: data: {}".format(rank, data))

win.Free()

separate()
