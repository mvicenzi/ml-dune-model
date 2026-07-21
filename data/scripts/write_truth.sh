#!/bin/bash

# Make box around text @climagic
function box() { t="$1xxxx";c=${2:-=}; echo ${t//?/$c}; echo "$c $1 $c"; echo ${t//?/$c}; }

# truth_files.txt is the list of all *trackid_map.h5 files in the dataset
# that is used to run the labelling algorithm in classify_pixels_sparse.py
# process truth file list into 40 batches and process them in parallel
# this works well on WCWC, be careful on other machines because this hits the I/O pretty hard
split -nl/40 -d truth_files.txt truth_batch

if [[ ! -d "logs" ]]; then
  mkdir logs
fi
for batch in truth_batch*; do
  box "$batch" && while read f; do python3 classify_pixels_sparse.py "$f"; done < "$batch" > "logs/logs_$batch.log" 2>&1 &
done

# wait for all processes to finish
wait
# cleanup
rm truth_batch*
