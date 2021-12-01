import sys, os
from subprocess import getoutput
import numpy as np
from mpi4py import MPI

rootname = 'Sband'
maxDM = 80 #max DM to search
Nsub = 32
os.chdir('/home/download/TestData2/subbands') #Absolute path, please modify it manually.
log = open('dedisperse.log', 'a')
#load dm list
dmlist = np.load("dmlist.npy")
dDM = float(sys.argv[1])
NDMs = int(sys.argv[2])
Nout = int(sys.argv[3])
datdownsamp = int(sys.argv[4])
subdownsamp = int(sys.argv[5])
filename = sys.argv[6]
if len(sys.argv) > 7:
    maskfile = sys.argv[7]
else:
    maskfile = None

def my_prepsubband_1(subDM):
    if maskfile:
        prepsubband = "prepsubband -sub -subdm %.2f -nsub %d -downsamp %d -mask ../%s -o %s %s" % (
        subDM, Nsub, subdownsamp, maskfile, rootname, filename)
    else:
        prepsubband = "prepsubband -sub -subdm %.2f -nsub %d -downsamp %d -o %s %s" % (
        subDM, Nsub, subdownsamp, rootname, filename)
    log.write(prepsubband)
    output = getoutput(prepsubband)
    return output
def my_prepsubband_2(lodm, subDM):
    subnames = rootname + "_DM%.2f.sub[0-9]*" % subDM
    prepsubcmd = "prepsubband -nsub %(Nsub)d -lodm %(lowdm)f -dmstep %(dDM)f -numdms %(NDMs)d -numout %(Nout)d -downsamp %(DownSamp)d -o %(root)s %(subfile)s" % {
        'Nsub': Nsub, 'lowdm': lodm, 'dDM': dDM, 'NDMs': NDMs, 'Nout': Nout, 'DownSamp': datdownsamp,
        'root': rootname, 'subfile': subnames}
    log.write(prepsubcmd)
    output = getoutput(prepsubcmd)
    return output


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numjobs = len(dmlist)

job_content = []  # the collection of jobs
for i, dm in enumerate(dmlist):
    lodm = dm[0]
    subDM = np.mean(dm)
    job_content.append((lodm, subDM))

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
work_content = [job_content[x] for x in this_worker_job]


for lodm, subDM in work_content:
    o_1 = my_prepsubband_1(subDM)
    log.write(o_1)
    o_2 = my_prepsubband_2(lodm, subDM)
    log.write(o_2)
log.close()
