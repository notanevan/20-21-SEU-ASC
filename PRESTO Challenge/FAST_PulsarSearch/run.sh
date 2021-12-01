#DATE=`date '+%F_%H:%M'`
rm -rf subbands
(time python ./my_pipeline.py ${1} ${2}) > log.pulsar_search 2>&1

