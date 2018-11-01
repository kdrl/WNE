#!/bin/bash
set -e
CORPUS="../data/sample_processed.txt"
NGRAM="../data/ngram_frequency.csv"
st="1e-7"
ep="1e-7"
make
./main --corpus_path=$CORPUS \
       --ngram_count_path=$NGRAM \
       --max_ngram_size=4 \
       --n_core=8 \
       --support_threshold=$st \
       --epsilon=$ep
