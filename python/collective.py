from mpi4py import MPI
import numpy as np

W = MPI.COMM_WORLD

rank = W.rank
size = W.size

def separate():
    W.barrier()
    if rank == 0:
        print("", flush = True)
    W.barrier()

print("{}: Before barrier".format(rank))

W.barrier()

print("{}: After barrier".format(rank))

separate()

if rank == 0:
    data = np.arange(10, dtype = 'i')
else:
    data = np.empty(10, dtype = 'i')

W.Bcast([data, MPI.INT], root = 0)

print("{}: Broadcasted: {}".format(rank, data), flush = True)

separate()

if rank == 0:
    data = np.empty([size, 10], dtype = 'i')
    data.T[:,:] = range(size)

buf = np.empty(10, dtype = 'i')
W.Scatter(data, buf)

print("{}: After scatter {}".format(rank, buf), flush = True)

separate()

buf += 100

W.Gather(buf, data)

if rank == 0:
    print("{}: After gather {}".format(rank, data), flush = True)

separate()

data = np.empty([size, 10], dtype = 'i')
buf = np.empty(10, dtype = 'i')

buf.fill(rank)

W.Allgather(buf, data)

print("{}: After allgather {}".format(rank, data), flush = True)

