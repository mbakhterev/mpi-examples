from mpi4py import MPI
import numpy as np

W = MPI.COMM_WORLD

rank = W.rank

def separate():
    W.barrier()
    if rank == 0:
        print("", flush = True)
    W.barrier()

data = np.zeros(10, dtype = 'i')
data[0] = rank

r = np.zeros(1, dtype = 'i')

W.Reduce(data, r, root = 0, op = MPI.SUM)

if rank == 0:
    print("{}: Reduce {}".format(rank, r), flush = True)

separate()
