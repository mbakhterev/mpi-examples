#!/usr/bin/env -S LD_PRELOAD=/usr/lib64/libslurm.so python 

from mpi4py import MPI

comm = MPI.COMM_WORLD

print("Hello world from %d of %d" % (comm.Get_rank(), comm.Get_size()))
