/*
    Embedding word-like n-grams
*/
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <chrono>

#include "cmdline.h"
#include "skipgram.h"

int main(int argc, char* argv[]) {

  // handling wide string
  std::ios_base::sync_with_stdio(false);
  std::locale default_loc("en_US.UTF-8");
  std::locale::global(default_loc);
  std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype); 
  std::wcout.imbue(ctype_default);
  std::wcin.imbue(ctype_default);
    
  // parsing parameters with https://github.com/tanakh/cmdline
  cmdline::parser a;
  a.add<std::string>("corpus_path", '\0', "corpus path", true);
  a.add<std::string>("word_data_path", '\0', "word_data_path", false);
  a.add<std::string>("ngram_data_path", '\0', "ngram_data_path", false);
  a.add<std::string>("output_path", '\0', "output path", true);

  a.add<int64_t>("size_window", '\0', "size_window", true);
  a.add<int64_t>("dim_embedding", '\0', "dim_embedding", true);
  a.add<int64_t>("seed", '\0', "seed", true);
  a.add<int64_t>("n_iteration", '\0', "n_iteration", true);
  a.add<int64_t>("n_negative_sample", '\0', "n_negative_sample", true);
  a.add<int64_t>("n_cores", '\0', "n_cores", true);

  a.add<double>("learning_rate", '\0', "learning_rate", true);
  a.add<double>("rate_sample", '\0', "rate_sample", true);
  a.add<double>("power_unigram_table", '\0', "power_unigram_table", true);

  a.add<int64_t>("embed_num", '\0', "embed_num", true);
  a.parse_check(argc, argv);

  std::string corpus_path = a.get<std::string>("corpus_path");
  std::string word_data_path = a.get<std::string>("word_data_path");
  std::string ngram_data_path = a.get<std::string>("ngram_data_path");
  std::string output_path = a.get<std::string>("output_path");

  int64_t size_window = a.get<int64_t>("size_window");
  int64_t dim_embedding = a.get<int64_t>("dim_embedding");
  int64_t seed = a.get<int64_t>("seed");
  int64_t n_iteration = a.get<int64_t>("n_iteration");
  int64_t n_negative_sample = a.get<int64_t>("n_negative_sample");
  int64_t n_cores = a.get<int64_t>("n_cores");

  double learning_rate = a.get<double>("learning_rate");
  double rate_sample = a.get<double>("rate_sample");
  double power_unigram_table = a.get<double>("power_unigram_table");

  int64_t embed_num = a.get<int64_t>("embed_num");

  // Load corpus
  std::wifstream fin_corpus(corpus_path);
  if (!fin_corpus.is_open()) {
    std::cout << "Invalid file name." << std::endl;
    return 0;
  }
  std::wstringstream wss;
  wss << fin_corpus.rdbuf();
  std::wstring corpus = wss.str();
  fin_corpus.close();

  // Load extracted words data
  std::wifstream fin_word(word_data_path);
  if (!fin_word.is_open()) {
    std::cout << "Invalid file name." << std::endl;
    return 0;
  }
  std::wstringstream wss2_w;
  wss2_w << fin_word.rdbuf();
  std::vector<std::wstring> vocabulary;
  std::unordered_map<std::wstring, int64_t> vocabulary2id_tmp;
  std::wstring linedata_w;
  int64_t id = 0;
  int64_t pos_w;
  std::wstring n_w;
  std::wstring delim_w = L"\t";
  while(embed_num && std::getline(wss2_w, linedata_w)){
    embed_num--;
    pos_w = linedata_w.find(delim_w);
    n_w = linedata_w.substr(0, pos_w);
    vocabulary.push_back(n_w);
    vocabulary2id_tmp[n_w] = id;
    id++;
  }
  fin_word.close();
  assert(vocabulary.size() == id);
    
  // Load extracted n-grams data
  std::wifstream fin_ngram(ngram_data_path);
  if (!fin_ngram.is_open()) {
    std::cout << "Invalid file name." << std::endl;
    return 0;
  }
  std::wstringstream wss2;
  wss2 << fin_ngram.rdbuf();
  int64_t* count_vocabulary_tmp = new int64_t[vocabulary.size()];
  // initialize the occuerrence of words with 1
  for(int64_t i = 0; i < vocabulary.size(); i++){
    count_vocabulary_tmp[i] = 1;
  }
  std::wstring linedata;
  int64_t pos;
  std::wstring n;
  std::wstring delim = L"\t";
  while(std::getline(wss2, linedata)){
    pos = linedata.find(delim);
    n = linedata.substr(0, pos);
    std::wistringstream wstrm(linedata.substr(pos+delim.length(), std::wstring::npos));
    int64_t number;
    wstrm >> number;
    if (vocabulary2id_tmp.find(n) != vocabulary2id_tmp.end()) count_vocabulary_tmp[vocabulary2id_tmp[n]] = number;
  }
  fin_ngram.close();
  std::vector<int64_t> count_vocabulary(count_vocabulary_tmp, &count_vocabulary_tmp[vocabulary.size()]);
  assert(count_vocabulary.size() == vocabulary.size());

  // Word embedding
  SkipGram sg(corpus, vocabulary, count_vocabulary,
              size_window, dim_embedding, seed,
              n_iteration, n_negative_sample, n_cores,
              learning_rate, rate_sample, power_unigram_table);
  auto t1 = std::chrono::high_resolution_clock::now();
  sg.train();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Training took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds\n";
  sg.save_vector(output_path);

  return 0;
}
