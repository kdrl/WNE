/*
    Count ngram frequency with lossy counting algorithm
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

#include "cmdline.h"
#include "lossycounting.h"

int main(int argc, char* argv[]) {

  // handling wide string
  std::ios_base::sync_with_stdio(false);
  std::locale default_loc("en_US.UTF-8");
  std::locale::global(default_loc);
  std::locale ctype_default(std::locale::classic(), default_loc, std::locale::ctype);
  std::wcout.imbue(ctype_default);
  std::wcin.imbue(ctype_default);

  // parsing parameters https://github.com/tanakh/cmdline
  cmdline::parser a;
  a.add<std::string>("corpus_path", '\0', "corpus path", true);
  a.add<std::string>("ngram_count_path", '\0', "ngram_count_path", true);
  a.add<std::string>("ngram_count_top_path", '\0', "ngram_count_top_path", false);
  a.add<int64_t>("extract_num", 0, "extract_num", false);
  a.add<int64_t>("max_ngram_size", '\0', "max_ngram_size", true);
  a.add<int64_t>("n_core", '\0', "n_core", true);
  a.add<double>("support_threshold", '\0', "support threshold", true);
  a.add<double>("epsilon", '\0', "epsilon", true);
  a.parse_check(argc, argv);
  std::string corpus_path = a.get<std::string>("corpus_path");
  std::string ngram_count_path = a.get<std::string>("ngram_count_path");
  std::string ngram_count_top_path = a.get<std::string>("ngram_count_top_path");
  int64_t extract_num = a.get<int64_t>("extract_num");
  int64_t max_ngram_size = a.get<int64_t>("max_ngram_size");
  int64_t n_core = a.get<int64_t>("n_core");
  double support_threshold = a.get<double>("support_threshold");
  double epsilon = a.get<double>("epsilon");

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

  //Extract frequently-used n-grams using lossy counting algorithm
  LossyCountingNgram counter(corpus, max_ngram_size, support_threshold, epsilon, n_core);
  counter.count_ngram();
  counter.extract_all_ngram_to_csv(ngram_count_path);
  if (extract_num != 0) {
    counter.extract_top_ngram_to_csv(ngram_count_top_path, extract_num);
  }

  return 0;
}
