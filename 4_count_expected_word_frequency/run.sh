#!/bin/bash
set -e
K=100000
CORPUS="../data/sample_processed.txt"
BOUNDARY="../data/word_boundary.hdf5"
WORD="../data/expected_word_frequency_top_$K.csv"
make
./main  --corpus_path=$CORPUS \
        --boundary_path=$BOUNDARY \
        --word_count_top_path=$WORD \
        --max_word_length=4 \
        --extract_num=$K \
        --n_core=8
