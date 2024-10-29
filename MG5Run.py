import multiprocessing
import socket

hostname = socket.gethostname()
nproc = multiprocessing.cpu_count()
if 'phy.pku.edu.cn' in hostname: nproc //= 2
pool = multiprocessing.Pool(nproc)
