#!/bin/bash

# Make box around text @climagic
function box() { t="$1xxxx";c=${2:-=}; echo ${t//?/$c}; echo "$c $1 $c"; echo ${t//?/$c}; }

idir="$1"
odir="$2"
max_files="$3"

cdir=`pwd`
cd "$idir"
filelist=`find . -name "*.tgz"`

count=0
for f in $filelist; do
  if [[ ! -z "$max_files" && "$count" -ge "$max_files" ]]; then
    box "Terminating after "$max_files" files.."
    break
  fi
  new_dir=`dirname $f | sed -e 's/\.\///g'`
  filename=`basename $f | sed -e 's/\.tgz//g' | sed -e 's/\.\///g'`
  echo "$filename"
  if [[ ! -d "$odir"/"$new_dir"/"$filename" ]]; then
    mkdir -p "$odir"/"$new_dir"/"$filename"
    # strip the "sparse folder" inside the archive and untar to our new folder
    tar -zxf "$f" --strip-components=1 -C "$odir"/"$new_dir"/"$filename"
    ((count++))
    echo "$odir"/"$new_dir"/"$filename"/*trackid_pid_map.h5 >> "$cdir"/truth_files.txt
  else
    echo "Folder already exists. Skipping.."
  fi
done
cd "$cdir"
box "Done!"
