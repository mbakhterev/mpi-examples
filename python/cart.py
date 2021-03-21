from mpi4py import MPI
import numpy as np
import sys

W = MPI.COMM_WORLD

print("{}: Starting with world size {}".format(W.rank, W.size), flush = True)

rows = int(np.floor(np.sqrt(W.size))) 
cols = W.size // rows

rank = W.rank
size = W.size

def separate(C):
    C.barrier()
    if rank == 0:
        print("", flush = True) 
    C.barrier()

C = W.Create_cart((rows, cols), periods = (False, True), reorder = True)

separate(C)

coord = C.coords

print("{}: My coordinates in grid: {}".format(rank, coord), flush = True)

separate(C)

h_src_dst = C.Shift(0, 1)
print("{}: HORIZONTAL neighbours: {}".format(rank, h_src_dst), flush = True)

separate(C)

v_src_dst = C.Shift(1, 1)
print("{}: VERTICAL neighbours: {}".format(rank, v_src_dst), flush = True)
