#ifndef SKIPGRAM_H
#define SKIPGRAM_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <locale>
#include <codecvt>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <thread>
#include <mutex>
#include <vector>

#include "cheaprand.h"

#define SIZE_TABLE_UNIGRAM 1000000
#define SIZE_CHUNK_PROGRESSBAR 1000

class SkipGram {
private:
  const std::wstring corpus;
  const std::vector<std::wstring> vocabulary;
  const std::vector<int64_t> count_vocabulary;

  const int64_t size_window;
  const int64_t dim_embedding;
  const int64_t seed;
  const int64_t n_iteration;
  const int64_t n_negative_sample;
  const int64_t n_cores;

  const double learning_rate;
  const double rate_sample;
  const double power_unigram_table;
 

  int64_t size_vocabulary;
  int64_t sum_count_vocabulary;
  int64_t max_length_word;
  std::unordered_map<std::wstring, int64_t> vocabulary2id;

  CheapRand cheaprand;
  int64_t* table_unigram;
  double* embeddings_words;
  double* embeddings_contexts_left;
  double* embeddings_contexts_right;

public:
  SkipGram(const std::wstring& _corpus,
           const std::vector<std::wstring>& _vocabulary,
           const std::vector<int64_t>& _count_vocabulary,
           const int64_t _size_window,
           const int64_t _dim_embedding,
           const int64_t _seed,
           const int64_t _n_iteration,
           const int64_t _n_negative_sample,
           const int64_t _n_cores,
           const double _learning_rate,
           const double _rate_sample,
           const double _power_unigram_table);
  ~SkipGram();
  void train();
  void save_vector(const std::string output_path);

private:
  void train_model_eachthread(const int64_t id_thread,
                              const int64_t i_wstr_start,
                              const int64_t length_str,
                              const int64_t n_cores);
  void initialize_parameters();
  void construct_unigramtable(const double power_unigram_table);
};
#endif
