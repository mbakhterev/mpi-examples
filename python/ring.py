from mpi4py import MPI
import numpy as np

W = MPI.COMM_WORLD

rank = W.rank
size = W.size

def separate(C):
    C.barrier()
    if rank == 0:
        print("", flush = True)
    C.barrier()

def next(n):
    return (n + 1) % size

def prev(n):
    return (n - 1) % size

if rank == 0:
    print("{}: Initiating. Manual. Synchronous. Values".format(rank))

separate(W)

if rank != 0:
    v = W.recv(source = prev(rank), tag = 12)
    print("{}: Received: {}".format(rank, v))

    W.send(v, dest = next(rank), tag = 12)
else:
    W.send("Hello ring world", dest=next(rank), tag=12)

    v = W.recv(source = prev(rank), tag = 12)
    print("{}: Received: {}".format(rank, v))

separate(W)

if rank == 0:
    print("{}: Initiating. Manual. Asynchronous. Values".format(rank))

separate(W)

rq = W.irecv(source = prev(rank))

if rank != 0:
    v = rq.wait()
    print("{}: Received: {}".format(rank, v))

    srq = W.isend(v, dest = next(rank))
else:
    srq = W.isend("Hello asynchronous world", dest = next(rank))

    v = rq.wait()
    print("{}: Received: {}".format(rank, v))

srq.wait()

separate(W)

if rank == 0:
    print("{}: Initiating. Auto. Asynchronous. Arrays".format(rank))

C = W.Create_cart([size], periods = (True), reorder = True)

src, dst = C.Shift(0, 1)

separate(C)

data = np.empty(10, dtype = 'i')
rq = C.Irecv([data, MPI.INT], source = src)

rank = C.rank

if rank != 0:
    rq.wait()
    print("{}: Received: {}".format(rank, data))

    srq = C.Isend([data, MPI.INT], dest = dst)
else:
    initial = np.arange(10, dtype = 'i')
    srq = C.Isend([initial, MPI.INT], dest = dst)

    rq.wait()
    print("{}: Received: {}".format(rank, data))

srq.wait()

separate(W)
