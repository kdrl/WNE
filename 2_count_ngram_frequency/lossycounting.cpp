#include "lossycounting.h"

LossyCountingNgram::LossyCountingNgram(const std::wstring& _corpus,
                                       const int64_t _max_ngram_size,
                                       const double _support_threshold,
                                       const double _epsilon,
                                       const int64_t _n_cores)
  : corpus(_corpus),
    max_ngram_size(_max_ngram_size),
    support_threshold(_support_threshold),
    epsilon(_epsilon),
    n_cores(_n_cores)
{
  assert(epsilon > 0);
  assert(support_threshold > 0);

  corpus_length = corpus.size();
  for (int64_t n = 1; n <=max_ngram_size; n++) {
      ngram_size_list.push_back(n);
  }
  bucket_size = static_cast<int64_t>(1.0 / epsilon);
  occurence_lower_bound = static_cast<int64_t>(support_threshold * corpus_length);
}

LossyCountingNgram::~LossyCountingNgram() {}

void LossyCountingNgram::count_ngram()
{
  const int64_t n_jobs = ngram_size_list.size();
  std::vector<std::thread> vector_threads(n_jobs);

  for (int64_t i_cores=0; i_cores<n_jobs; i_cores++) {
    if (i_cores >= n_cores) vector_threads.at(i_cores-n_cores).join();
    vector_threads.at(i_cores) = std::thread(&LossyCountingNgram::count_ngram_each, this, ngram_size_list[i_cores]);
  }

  //wait for thread left to complete
  for (auto& th : vector_threads) if (th.joinable()) th.join();

  // Sort all
  std::sort(counted_data.begin(), counted_data.end(),
            [](const std::pair<std::wstring, int64_t>& lhs,
               const std::pair<std::wstring, int64_t>& rhs)
            { return lhs.second > rhs.second; });
}

void LossyCountingNgram::count_ngram_each(const int64_t ngram_size)
{
  std::unordered_map<std::wstring, int64_t> counter_lossycounting, error_lossycounting;
  std::wstring key, ngram;
  int64_t i_bucket = 1;

  for (int64_t i=0; i <= corpus_length-ngram_size; i++) {

    ngram = corpus.substr(i, ngram_size);

    // If `ngram` exists in counter
    if (counter_lossycounting.find(ngram) != counter_lossycounting.end()) {
      counter_lossycounting[ngram] += 1;
    } else {
      counter_lossycounting.insert(std::make_pair(ngram, 1));
      error_lossycounting.insert(std::make_pair(ngram, i_bucket - 1));
    }

    if (i && i % bucket_size == 0) {
      std::vector<std::wstring> vocabulary_current(counter_lossycounting.size());
      for (auto& elem : counter_lossycounting) {
        vocabulary_current.push_back(elem.first);
      }
      for (std::wstring& key : vocabulary_current) {
        if (counter_lossycounting[key] + error_lossycounting[key] <= i_bucket) {
          counter_lossycounting.erase(key);
          error_lossycounting.erase(key);
        }
      }
      i_bucket += 1;
    }

  }

  std::vector<std::pair<std::wstring, int64_t>> elems(counter_lossycounting.begin(),
                                                      counter_lossycounting.end());
  std::sort(elems.begin(), elems.end(),
            [](const std::pair<std::wstring, int64_t>& lhs,
               const std::pair<std::wstring, int64_t>& rhs)
            { return lhs.second > rhs.second; });

  std::vector<std::pair<std::wstring, int64_t>> counted_data_eachthread;

  for (int64_t i=0; i<elems.size(); i++) {
    // Condition : (ngram's occurences >= lower_bound)
    if (elems[i].second >= occurence_lower_bound) counted_data_eachthread.push_back(elems[i]);
  }

  // Update ngrams & counts
  std::lock_guard<std::mutex> lock(mtx);
  counted_data.insert(counted_data.end(), counted_data_eachthread.begin(), counted_data_eachthread.end());
}

void LossyCountingNgram::extract_all_ngram_to_csv(const std::string ngram_count_path)
{
  std::string output_path = ngram_count_path;
  std::cout << "Saving counted ngrams to " << output_path << std::endl;
  std::wofstream fout(output_path);
  for (int64_t i=0; i<counted_data.size(); i++) {
    fout << counted_data[i].first << "\t" << counted_data[i].second << std::endl;
    if (i) assert(counted_data[i-1].second >= counted_data[i].second);
  }
  fout.close();
  std::cout << "Done" << std::endl;
}

void LossyCountingNgram::extract_all_ngram(std::unordered_map<std::wstring, int64_t>& placeholder)
{
  for (auto it : counted_data) {
      placeholder.insert(it);
  }
}

void LossyCountingNgram::extract_top_ngram_to_csv(const std::string ngram_count_top_path, const int64_t extract_num)
{
  std::string output_path = ngram_count_top_path;
  std::cout << "Extract top-" << extract_num << " ngrams to " << output_path << std::endl;

  std::unordered_map<std::wstring, int64_t> extracted_ngram_map;
  int64_t num = 0;

  for (int64_t i=0; i<counted_data.size(); i++) {
      if (extracted_ngram_map.find(counted_data[i].first) == extracted_ngram_map.end()) {
         extracted_ngram_map.insert(counted_data[i]);
         num++;
         if (num == extract_num) break;
      }
  }
  if(num < extract_num){
    std::cout << std::endl << "[WARNING] Not enough ngram counted compared to extract_num" << std::endl << std::endl;
  }
  std::cout << "Total " << num << " ngrams extracted" << std::endl;

  assert(num == extracted_ngram_map.size());
  assert(num <= extract_num);

  std::vector<std::pair<std::wstring, int64_t>> extracted_ngram_vector(extracted_ngram_map.begin(),
                                                                       extracted_ngram_map.end());
  std::sort(extracted_ngram_vector.begin(), extracted_ngram_vector.end(),
            [](const std::pair<std::wstring, int64_t>& lhs,
               const std::pair<std::wstring, int64_t>& rhs)
            { return lhs.second > rhs.second; });

  std::wofstream fout(output_path);
  for (int64_t i=0; i<extracted_ngram_vector.size(); i++) {
    fout << extracted_ngram_vector[i].first << "\t" << extracted_ngram_vector[i].second << std::endl;
    if (i) assert(extracted_ngram_vector[i-1].second >= extracted_ngram_vector[i].second); // Check Sort
  }
  fout.close();

  std::cout << "Done" << std::endl;
}

void LossyCountingNgram::extract_top_ngram(std::vector<std::wstring>& vocabulary, std::vector<int64_t>& count_vocabulary, const int64_t extract_num)
{

  std::cout << "Extract top-" << extract_num << " ngrams" << std::endl;

  std::unordered_map<std::wstring, int64_t> extracted_ngram_map;
  int64_t num = 0;

  for (int64_t i=0; i<counted_data.size(); i++) {
      if (extracted_ngram_map.find(counted_data[i].first) == extracted_ngram_map.end()) {
         extracted_ngram_map.insert(counted_data[i]);
         num++;
         if (num == extract_num) break;
      }
  }
  if(num < extract_num){
    std::cout << std::endl << "[WARNING] Not enough ngram counted compared to extract_num" << std::endl << std::endl;
  }
  std::cout << "Total " << num << " ngrams extracted" << std::endl;

  assert(num == extracted_ngram_map.size());
  assert(num <= extract_num);

  std::vector<std::pair<std::wstring, int64_t>> extracted_ngram_vector(extracted_ngram_map.begin(),
                                                                       extracted_ngram_map.end());
  std::sort(extracted_ngram_vector.begin(), extracted_ngram_vector.end(),
            [](const std::pair<std::wstring, int64_t>& lhs,
               const std::pair<std::wstring, int64_t>& rhs)
            { return lhs.second > rhs.second; });

  for (auto it : extracted_ngram_vector) { // Copy sorted elements
      vocabulary.push_back(it.first);
      count_vocabulary.push_back(it.second);
  }

  std::cout << "Done" << std::endl;
}
