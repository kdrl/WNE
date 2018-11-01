#ifndef COUNTINGWORD_H
#define COUNTINGWORD_H

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
#include <unordered_set>
#include <numeric>
#include <vector>
#include <thread>
#include <mutex>

class CountingWord
{
  private:
    const std::wstring corpus;
    const std::vector<double> boundary_data;
    const int64_t max_word_length;
    const int64_t extract_num_maximun;
    const int64_t n_cores;

    int64_t corpus_length;
    std::vector<int64_t> word_length_list;
    std::vector<std::pair<std::wstring, double>> counted_data;
    std::mutex mtx;

  public:
    CountingWord(const std::wstring& _corpus,
                 const std::vector<double> _boundary_data,
                 const int64_t _max_word_length,
                 const int64_t _extract_num_maximun,
                 const int64_t _n_cores);
    ~CountingWord();
    void count_word();
    void count_word_each(const int64_t word_length);
    void extract_all_word_to_csv(const std::string word_count_path);
    void extract_top_word_to_csv(const std::string word_count_top_path, const int64_t extract_num);
};

#endif
