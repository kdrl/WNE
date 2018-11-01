/*
    Count expected word frequency and extract word-like ngrams
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

#include "H5Cpp.h"
#include "cmdline.h"
#include "counting_word.h"

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
  a.add<std::string>("boundary_path", '\0', "boundary_path", true);
  a.add<std::string>("word_count_top_path", '\0', "word_count_top_path", true);
  a.add<int64_t>("max_word_length", '\0', "max_word_length", true);
  a.add<int64_t>("extract_num", '\0', "extract_num", true);
  a.add<int64_t>("n_core", '\0', "n_core", true);
  a.parse_check(argc, argv);
  std::string corpus_path = a.get<std::string>("corpus_path");
  std::string boundary_path = a.get<std::string>("boundary_path");
  std::string word_count_top_path = a.get<std::string>("word_count_top_path");
  int64_t max_word_length = a.get<int64_t>("max_word_length");
  int64_t extract_num = a.get<int64_t>("extract_num");
  int64_t n_core = a.get<int64_t>("n_core");

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

  // Load boundary data
  // hardwire data properties
  const int H5ERROR = 11;
  const H5std_string FILENAME = boundary_path;
  const H5std_string WORDBOUNDARY = "word_boundary";
  const int NDIMS = 1;
  hsize_t dims[1];
  // open file
  H5::H5File file = H5::H5File(FILENAME, H5F_ACC_RDONLY);
  // get signal dataset
  H5::DataSet signal_dset = file.openDataSet(WORDBOUNDARY);
  // check that signal is float
  if (signal_dset.getTypeClass() != H5T_FLOAT) {
    std::cerr << "signal dataset has wrong type" << std::endl;
    return H5ERROR;
  }
  // check that signal is double
  if (signal_dset.getFloatType().getSize() != sizeof(double)) {
    std::cerr << "signal dataset has wrong type size" << std::endl;
    return H5ERROR;
  }
  // get the dataspace
  H5::DataSpace signal_dspace = signal_dset.getSpace();
  // check that signal has 2 dims
  if (signal_dspace.getSimpleExtentNdims() != NDIMS) {
    std::cerr << "signal dataset has wrong number of dimensions"
          << std::endl;
    return H5ERROR;
  }
  // get dimensions
  signal_dspace.getSimpleExtentDims(dims, NULL);
  // allocate memory and read data
  double* boundary_data_tmp;
  boundary_data_tmp = new double[(int)(dims[0])];
  H5::DataSpace signal_mspace(NDIMS, dims);
  signal_dset.read(boundary_data_tmp, H5::PredType::NATIVE_DOUBLE, signal_mspace, signal_dspace);
  // all done with file
  file.close();

  // check and convert
  assert((int)(dims[0]) == corpus.length());
  std::vector<double> boundary_data(boundary_data_tmp, &boundary_data_tmp[(int)(dims[0])]);

  CountingWord wordcounter(corpus, boundary_data, max_word_length, extract_num, n_core);
  wordcounter.count_word();
  wordcounter.extract_top_word_to_csv(word_count_top_path, extract_num);

  return 0;
}
