from mpi4py import MPI
import numpy as np

W = MPI.COMM_WORLD

rank = W.rank

def separate():
    W.barrier()
    if rank == 0:
        print("", flush = True)
    W.barrier()

# Vectors

if rank == 0:
    print("Vectors", flush = True)

separate()

data = rank * np.arange(10, dtype = 'i')

print("{}: data: {}".format(rank, data), flush = True)

separate()

r = np.empty(10, dtype = 'i')

W.Reduce(data, r, root = 0, op = MPI.SUM)

if rank == 0:
    print("{}: result: {}".format(rank, r), flush = True)

separate()

# Matrix 

if rank == 0:
    print("Matrices", flush = True)

separate()

data = rank * np.arange(9, dtype = 'i').reshape(3, 3)

print("{}: data: {}".format(rank, data), flush = True)

separate()

r = np.empty(9, dtype = 'i').reshape(3, 3)

W.Reduce(data, r, root = 2, op = MPI.SUM)

if rank == 2:
    print("{}: result {}".format(rank, r), flush = True)

separate()

# Allreduce and order negotiation 

if rank == 0:
    print("Allreduce", flush = True)

separate()

W.Allreduce(data, r, op = MPI.MAX)

for i in range(0, W.size):
    W.barrier()
    if i == rank:
        print("{}: result: {}".format(rank, r), flush = True)
        
separate()
