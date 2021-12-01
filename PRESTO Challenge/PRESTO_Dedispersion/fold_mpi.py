import sys, os, glob, re
from subprocess import getoutput
from presto import sifting
import numpy as np
from mpi4py import MPI
from operator import attrgetter

cwd = os.getcwd()
os.chdir('/home/download/TestData1/subbands') #Absolute path, please modify it manually.
logfile = open('folding.log', 'wt')

Nint = int(sys.argv[1])
Nsub = int(sys.argv[2])
filename = sys.argv[3]
rootname = sys.argv[4]
zmax = 0

#load candicates file
cands = np.load("cands.npy", allow_pickle=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numjobs = len(cands)
# arrange the works and jobs
if rank == 0:
    # this is head worker
    # jobs are arranged by this worker
    job_all_idx = list(range(numjobs))
else:
    job_all_idx = None

job_all_idx = comm.bcast(job_all_idx, root=0)

njob_per_worker = int(numjobs / size)
# the number of jobs should be a multiple of the NumProcess[MPI]
this_worker_job = [job_all_idx[x] for x in range(rank * njob_per_worker, (rank + 1) * njob_per_worker)]
# map the index to parameterset [eps,anis]
work_content = [cands[x] for x in this_worker_job]

for cand in work_content:
    foldcmd = "prepfold -n %(Nint)d -nsub %(Nsub)d -dm %(dm)f -p %(period)f %(filfile)s -o %(outfile)s -noxwin -nodmsearch" % {
        'Nint': Nint, 'Nsub': Nsub, 'dm': cand.DM, 'period': cand.p, 'filfile': filename,
        'outfile': rootname + '_DM' + cand.DMstr}  # full plots
    logfile.write(foldcmd)
    output = getoutput(foldcmd)
    logfile.write(output)
logfile.close()


