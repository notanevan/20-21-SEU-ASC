"""
A simple pipelien for demostrating presto
Weiwei Zhu
2015-08-14
Max-Plank Institute for Radio Astronomy
zhuwwpku@gmail.com
"""
import os, sys, glob, re
from presto import sifting
from subprocess import getoutput
import numpy as np
from builtins import map
from operator import attrgetter
from multiprocessing import Pool
import traceback


rootname = 'Sband'
maxDM = 80 #max DM to search
Nsub = 32 #32 subbands
Nint = 64 #64 sub integration
Tres = 0.5 #ms
zmax = 0

filename = sys.argv[1]
if len(sys.argv) > 2:
    maskfile = sys.argv[2]
else:
    maskfile = None

print('''

====================Read Header======================

''')

readheadercmd = 'readfile %s' % filename
print(readheadercmd)
output = getoutput(readheadercmd)
print(output)
header = {}
for line in output.split('\n'):
    items = line.split("=")
    if len(items) > 1:
        header[items[0].strip()] = items[1].strip()


print('''

============Generate Dedispersion Plan===============

''')

try:
    Nchan = int(header['Number of channels'])
    tsamp = float(header['Sample time (us)']) * 1.e-6
    BandWidth = float(header['Total Bandwidth (MHz)'])
    fcenter = float(header['Central freq (MHz)'])
    Nsamp = int(header['Spectra per file'])

    ddplancmd = 'DDplan.py -d %(maxDM)s -n %(Nchan)d -b %(BandWidth)s -t %(tsamp)f -f %(fcenter)f -s %(Nsub)s -o DDplan.ps' % {
            'maxDM':maxDM, 'Nchan':Nchan, 'tsamp':tsamp, 'BandWidth':BandWidth, 'fcenter':fcenter, 'Nsub':Nsub}
    print(ddplancmd)
    ddplanout = getoutput(ddplancmd)
    print(ddplanout)
    planlist = ddplanout.split('\n')
    ddplan = []
    planlist.reverse()
    for plan in planlist:
        if plan == '':
            continue
        elif plan.strip().startswith('Low DM'):
            break
        else:
            ddplan.append(plan)
    ddplan.reverse()
except Exception as e:
    print('Exception:', str(e))
    print('failed at generating DDplan.')
    sys.exit(0)

print('''

================Dedisperse Subbands==================

''')
try:
    cwd = os.getcwd()
    if not os.access('subbands', os.F_OK):
        os.mkdir('subbands')
    os.chdir('subbands')
    logfile = open('dedisperse.log', 'wt')
    for line in ddplan:
        ddpl = line.split()
        lowDM = float(ddpl[0])
        hiDM = float(ddpl[1])
        dDM = float(ddpl[2])
        DownSamp = int(ddpl[3])
        NDMs = int(ddpl[6])
        calls = int(ddpl[7])
        Nout = Nsamp/DownSamp
        Nout -= (Nout % 500)
        dmlist = np.split(np.arange(lowDM, hiDM, dDM), calls)
        pros = len(dmlist)
        np.save("dmlist.npy", dmlist)

        subdownsamp = DownSamp/2
        datdownsamp = 2
        if DownSamp < 2: subdownsamp = datdownsamp = 1

        if maskfile:
            mpi_dd = "mpiexec -n %d -num-cores 32 python ../dd_MPI.py %f %d %d %d %d %s %s" % (
            pros, dDM, NDMs, Nout, datdownsamp, subdownsamp, '../' + filename, maskfile)
        else:
            mpi_dd = "mpiexec -n %d -num-cores 32 python ../dd_MPI.py %f %d %d %d %d %s" % (
            pros, dDM, NDMs, Nout, datdownsamp, subdownsamp, '../' + filename)
        print(mpi_dd)
        output = getoutput(mpi_dd)

    os.system('rm *.sub*')
    logfile.close()
    os.chdir(cwd)
except:
    #traceback.print_exc()
    print('failed at prepsubband.')
    os.chdir(cwd)
    sys.exit(0)

c = ''
def runcmds(cmds):
    traceback.print_exc()
    output = []
    for cmd in cmds:
        output.append(getoutput(cmd))
    return c.join(output)
print('''

================fft-search subbands==================

''')
try:
    commands = []
    os.chdir('subbands')
    datfiles = glob.glob("*.dat")
    logfile = open('fft.log', 'wt')

    threads = Pool()
    commands.clear()

    for df in datfiles:
        fftcmd = "realfft %s" % df
        print(fftcmd)
        commands.append([fftcmd])

    logfile.write(c.join(threads.map(runcmds, commands)))
    threads.map(runcmds, commands)
    threads.close()
    threads.join()

    logfile.close()
    logfile = open('accelsearch.log', 'wt')
    fftfiles = glob.glob("*.fft")

    threads = Pool()
    commands = []

    for fftf in fftfiles:
        searchcmd = "accelsearch -zmax %d %s" % (zmax, fftf)
        print(searchcmd)
        commands.append([searchcmd])
    c = ''
    logfile.write(c.join(threads.map(runcmds, commands)))
    threads.close()
    threads.join()

    logfile.close()
    os.chdir(cwd)
except:
    traceback.print_exc()
    print('failed at fft search.')
    os.chdir(cwd)
    sys.exit(0)


#"""

def ACCEL_sift(zmax):
    '''
    The following code come from PRESTO's ACCEL_sift.py
    '''

    globaccel = "*ACCEL_%d" % zmax
    globinf = "*DM*.inf"
    # In how many DMs must a candidate be detected to be considered "good"
    min_num_DMs = 2
    # Lowest DM to consider as a "real" pulsar
    low_DM_cutoff = 2.0
    # Ignore candidates with a sigma (from incoherent power summation) less than this
    sifting.sigma_threshold = 4.0
    # Ignore candidates with a coherent power less than this
    sifting.c_pow_threshold = 100.0

    # If the birds file works well, the following shouldn't
    # be needed at all...  If they are, add tuples with the bad
    # values and their errors.
    #                (ms, err)
    sifting.known_birds_p = []
    #                (Hz, err)
    sifting.known_birds_f = []

    # The following are all defined in the sifting module.
    # But if we want to override them, uncomment and do it here.
    # You shouldn't need to adjust them for most searches, though.

    # How close a candidate has to be to another candidate to
    # consider it the same candidate (in Fourier bins)
    sifting.r_err = 1.1
    # Shortest period candidates to consider (s)
    sifting.short_period = 0.0005
    # Longest period candidates to consider (s)
    sifting.long_period = 15.0
    # Ignore any candidates where at least one harmonic does exceed this power
    sifting.harm_pow_cutoff = 8.0

    #--------------------------------------------------------------

    # Try to read the .inf files first, as _if_ they are present, all of
    # them should be there.  (if no candidates are found by accelsearch
    # we get no ACCEL files...
    inffiles = glob.glob(globinf)
    candfiles = glob.glob(globaccel)
    # Check to see if this is from a short search
    if len(re.findall("_[0-9][0-9][0-9]M_", inffiles[0])):
        dmstrs = [x.split("DM")[-1].split("_")[0] for x in candfiles]
    else:
        dmstrs = [x.split("DM")[-1].split(".inf")[0] for x in inffiles]

    dms = list(map(float, dmstrs))

    dms.sort()
    dmstrs = ["%.2f"%x for x in dms]

    # Read in all the candidates
    cands = sifting.read_candidates(candfiles)

    # Remove candidates that are duplicated in other ACCEL files
    if len(cands):
        cands = sifting.remove_duplicate_candidates(cands)

    # Remove candidates with DM problems
    if len(cands):
        cands = sifting.remove_DM_problems(cands, min_num_DMs, dmstrs, low_DM_cutoff)

    # Remove candidates that are harmonically related to each other
    # Note:  this includes only a small set of harmonics
    if len(cands):
        cands = sifting.remove_harmonics(cands)

    # Write candidates to STDOUT
    if len(cands):
        cands.sort(key = attrgetter('sigma'), reverse = True)
        #for cand in cands[:1]:
            #print cand.filename, cand.candnum, cand.p, cand.DMstr
        #sifting.write_candlist(cands)
    return cands


print('''

================sifting candidates==================

''')

cwd = os.getcwd()
os.chdir('subbands')
cands = ACCEL_sift(zmax)
os.chdir(cwd)

print('''

================folding candidates==================

''')

try:
    cwd = os.getcwd()
    os.chdir('subbands')
    os.system('ln -sf ../%s %s' % (filename, filename))
    pros = len(cands)
    print(pros)
    np.save("cands.npy", cands)

    mpi_folding = "mpiexec -n %d -cpus-per-proc 1 -num-cores 32 python ../fold_mpi.py %d %d %s %s" % (pros, Nint, Nsub, filename, rootname)
    print(mpi_folding)
    output = getoutput(mpi_folding)
    print("ALL DONE")
    os.chdir(cwd)

except Exception as e:
    #traceback.print_exc()
    print('failed at folding candidates.')
    os.chdir(cwd)
    sys.exit(0)
