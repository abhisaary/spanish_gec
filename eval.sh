#!/bin/bash
# Call this following script as >>> sh eval.sh <path/to/eval/dir/>
# Generate m2 comparison files between each set of original - target and original - predicted files
declare -a data_splits=("train" "val" "test")
for data_split in "${data_splits[@]}"
do
  errant_parallel -orig "$1${data_split}_original.txt" -cor "$1${data_split}_target.txt" -out "$1${data_split}_reference.m2"
  errant_parallel -orig "$1${data_split}_original.txt" -cor "$1${data_split}_predicted.txt" -out "$1${data_split}_hypothesis.m2"
done

# Generate F-score for the each set of reference and hypothesis m2 files
declare -a data_splits=("train" "val" "test")
for data_split in "${data_splits[@]}"
do
  errant_compare -hyp "$1${data_split}_hypothesis.m2" -ref "$1${data_split}_reference.m2" -cs > "$1${data_split}_results-cs.txt"
done