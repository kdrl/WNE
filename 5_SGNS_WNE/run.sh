#!/bin/bash
set -e
K=100000
CORPUS="../data/sample_processed.txt"
NGRAM="../data/ngram_frequency.csv"
WORD="../data/expected_word_frequency_top_$K.csv"
OUTPUT="../data/embeddings.txt"
make
./main --corpus_path=$CORPUS \
       --ngram_data_path=$NGRAM \
       --word_data_path=$WORD \
       --output_path=$OUTPUT \
       --embed_num=$K \
       --size_window=1 \
       --dim_embedding=50 \
       --seed=2018 \
       --n_iteration=5 \
       --n_negative_sample=10 \
       --n_cores=8 \
       --learning_rate=0.05 \
       --rate_sample=0.0001 \
       --power_unigram_table=0.75   
