#include "counting_word.h"

CountingWord::CountingWord(const std::wstring& _corpus,
                           const std::vector<double> _boundary_data,
                           const int64_t _max_word_length,
                           const int64_t _extract_num_maximun,
                           const int64_t _n_cores)
  : corpus(_corpus),
    boundary_data(_boundary_data),
    max_word_length(_max_word_length),
    extract_num_maximun(_extract_num_maximun),
    n_cores(_n_cores)
{
  corpus_length = corpus.size();
  for (int64_t n = 1; n <=max_word_length; n++) {
      word_length_list.push_back(n);
  }
}

CountingWord::~CountingWord() {}

void CountingWord::count_word()
{
  const int64_t n_jobs = word_length_list.size();
  std::vector<std::thread> vector_threads(n_jobs);

  for (int64_t i_cores=0; i_cores<n_jobs; i_cores++) {
    if (i_cores >= n_cores) vector_threads.at(i_cores-n_cores).join();
    vector_threads.at(i_cores) = std::thread(&CountingWord::count_word_each, this, word_length_list[i_cores]);
  }

  //wait for thread left to complete
  for (auto& th : vector_threads) if (th.joinable()) th.join();

  // Sort all
  std::sort(counted_data.begin(), counted_data.end(),
            [](const std::pair<std::wstring, double>& lhs,
               const std::pair<std::wstring, double>& rhs)
            { return lhs.second > rhs.second; });
}

void CountingWord::count_word_each(const int64_t word_length)
{
  std::unordered_map<std::wstring, double> word_count_each;
  std::wstring word;
  double probability;

  for (int64_t i=0; i <= corpus_length-word_length; i++) {

    word = corpus.substr(i, word_length);

    probability = boundary_data[i];
    for(int64_t j = 1; j < word_length; j++){
        probability = probability * (1.0 - boundary_data[i+j]);
    }
    if (i+word_length < corpus_length) probability = probability * boundary_data[i+word_length];

    if (word_count_each.find(word) != word_count_each.end()) {
      word_count_each[word] += probability;
    } else {
      word_count_each.insert(std::make_pair(word, probability));
    }

  }

  std::vector<std::pair<std::wstring, double>> elems(word_count_each.begin(),
                                                     word_count_each.end());
  std::sort(elems.begin(), elems.end(),
            [](const std::pair<std::wstring, double>& lhs,
               const std::pair<std::wstring, double>& rhs)
            { return lhs.second > rhs.second; });

  int64_t min_num = (elems.size() < extract_num_maximun) ? elems.size() : extract_num_maximun;

  // Update
  std::lock_guard<std::mutex> lock(mtx);
  for (int64_t i=0; i<min_num; i++) {
    counted_data.push_back(elems[i]);
  }
}

void CountingWord::extract_all_word_to_csv(const std::string word_count_path){
  std::string output_path = word_count_path;
  std::cout << "Saving word-like ngrams to " << output_path << std::endl;
  std::wofstream fout(output_path);
  for (auto it : counted_data) {
    fout << it.first << "\t" << it.second << std::endl;
  }
  fout.close();
  std::cout << "Done" << std::endl;
}

void CountingWord::extract_top_word_to_csv(const std::string word_count_top_path, const int64_t extract_num)
{
  std::string output_path = word_count_top_path;
  std::cout << "Extract " << extract_num << " words to " << output_path << std::endl;

  std::unordered_map<std::wstring, double> extracted_word_map;
  int64_t num = 0;

  for (int64_t i=0; i<counted_data.size(); i++) {
      if (extracted_word_map.find(counted_data[i].first) == extracted_word_map.end()) {
         extracted_word_map.insert(counted_data[i]);
         num++;
         if (num == extract_num) break;
      }
  }
  if(num < extract_num){
    std::cout << std::endl << "[WARNING] Not enough words are saved compared to extract_num" << std::endl << std::endl;
  }
  std::cout << "Total " << num << " words extracted with word-probability-order" << std::endl;

  assert(num == extracted_word_map.size());
  assert(num <= extract_num);

  std::vector<std::pair<std::wstring, double>> extracted_word_vector(extracted_word_map.begin(),
                                                                     extracted_word_map.end());
  std::sort(extracted_word_vector.begin(), extracted_word_vector.end(),
            [](const std::pair<std::wstring, double>& lhs,
               const std::pair<std::wstring, double>& rhs)
            { return lhs.second > rhs.second; });

  std::wofstream fout(output_path);
  for (int64_t i=0; i<extracted_word_vector.size(); i++) {
    fout << extracted_word_vector[i].first << "\t" << extracted_word_vector[i].second << std::endl;
    if (i) assert(extracted_word_vector[i-1].second >= extracted_word_vector[i].second); // Check Sort
  }
  fout.close();

  std::cout << "Done" << std::endl;
}
