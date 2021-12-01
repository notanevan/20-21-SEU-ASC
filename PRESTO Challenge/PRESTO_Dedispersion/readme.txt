my_pipeline.py runs the pipeline of presto.

dd_MPI.py is a parallel script to compute dedispersion.
fold_mpi.py is a parallel script for folding operations.
Attention：In dd_MPI.py, fold_mpi.py, os.chdir() use absolute path, since using relative path causes error. You need to modify it.

run.sh is a shell that calls my_pipeline.py.

Please input it on terminal:
bash run.sh GBT_Lband_PSR.fil

Our log.pulsar_search is a little different because commands in dedisperse and folding are written into subbands/dedisperse.log and subbands/folding.log instead of log.pulsar_search.
