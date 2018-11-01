#include "skipgram.h"

SkipGram::SkipGram(const std::wstring& _corpus,
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
                   const double _power_unigram_table)
  : corpus(_corpus),
    vocabulary(_vocabulary),
    count_vocabulary(_count_vocabulary),
    size_window(_size_window),
    dim_embedding(_dim_embedding),
    seed(_seed),
    n_iteration(_n_iteration),
    n_negative_sample(_n_negative_sample),
    n_cores(_n_cores),
    learning_rate(_learning_rate),
    rate_sample(_rate_sample),
    power_unigram_table(_power_unigram_table)
{

  // Check given parameter
  assert(size_window >= 0);
  assert(dim_embedding > 0);
  assert(seed > 0);
  assert(n_iteration >= 0);
  assert(n_negative_sample >= 0);
  assert(n_cores >= 0);
  assert(learning_rate > 0);
  assert(rate_sample > 0);
  assert(power_unigram_table > 0);

  // Parameter setting
  size_vocabulary = vocabulary.size();
  sum_count_vocabulary = std::accumulate(count_vocabulary.begin(), count_vocabulary.end(), 0);
  max_length_word = 0;
  for (auto &v : vocabulary) {
    const int64_t length = v.size();
    if (length > max_length_word) max_length_word = length;
  }

  cheaprand = CheapRand(seed);

  for (int64_t i=0; i<size_vocabulary; i++) {
    vocabulary2id[vocabulary[i]] = i;
  }

  initialize_parameters();
  construct_unigramtable(power_unigram_table);

  std::wcout << std::endl;
  std::wcout << "###### SGNS-WNE ######" << std::endl;
  std::wcout << "corpus.size()       : " << corpus.size() << std::endl;
  std::wcout << "vocabulary.size()   : " << vocabulary.size() << std::endl;
  std::wcout << "size_window         : " << size_window << std::endl;
  std::wcout << "dim_embedding       : " << dim_embedding << std::endl;
  std::wcout << "seed                : " << seed << std::endl;
  std::wcout << "n_iteration         : " << n_iteration << std::endl;
  std::wcout << "n_negative_sample   : " << n_negative_sample << std::endl;
  std::wcout << "n_cores             : " << n_cores << std::endl;
  std::wcout << "learning_rate       : " << learning_rate << std::endl;
  std::wcout << "rate_sample         : " << rate_sample << std::endl;
  std::wcout << "power_unigram_table : " << power_unigram_table << std::endl;
  std::wcout << "max_length_word     : " << max_length_word << std::endl;
  std::wcout << "######################" << std::endl;
}

SkipGram::~SkipGram() {
  delete[] embeddings_words;
  delete[] embeddings_contexts_left;
  delete[] embeddings_contexts_right;
  delete[] table_unigram;
}

void SkipGram::initialize_parameters() {
  const int64_t n = size_vocabulary*dim_embedding;
  const double _min = -1.0/dim_embedding;
  const double _max =  1.0/dim_embedding;

  // Allocates memory for vector representations
  embeddings_words          = new double[n];
  embeddings_contexts_left  = new double[n];
  embeddings_contexts_right = new double[n];

  for (int64_t i=0; i<n; i++) {
    embeddings_words[i] = cheaprand.generate_rand_uniform(_min, _max);
    embeddings_contexts_left[i] = 0.0;
    embeddings_contexts_right[i] = 0.0;
  }
}

void SkipGram::train() {
  const int64_t length_corpus = corpus.size();
  const int64_t length_chunk = length_corpus / n_cores;
  int64_t i_corpus_start = 0;
  std::vector<std::thread> vector_threads(n_cores);

  for (int64_t id_thread=0; id_thread<n_cores; id_thread++) {
    vector_threads.at(id_thread) = std::thread(&SkipGram::train_model_eachthread,
                                               this,
                                               id_thread,
                                               i_corpus_start, length_chunk, n_cores);
    i_corpus_start += length_chunk;
  }

  for (int64_t id_thread=0; id_thread<n_cores; id_thread++) {
    vector_threads.at(id_thread).join();
  }
}

void SkipGram::train_model_eachthread(const int64_t id_thread,
                                      const int64_t i_corpus_start,
                                      const int64_t length_str,
                                      const int64_t n_cores)
{
  const std::wstring corpus_thread = corpus.substr(i_corpus_start, length_str);
  std::unordered_map<std::wstring, int64_t> vocabulary2id_thread = vocabulary2id;
  std::vector<int64_t> count_vocabulary_thread = count_vocabulary;

  double* gradient_words = new double[dim_embedding];
  CheapRand cheaprand_thread(id_thread + seed);

  if (id_thread == n_cores - 1) std::wcout << std::endl;

  for (int64_t i_iteration=0; i_iteration<n_iteration; i_iteration++) {
    // For each position in corpus
    for (int64_t i_str=0; i_str<length_str; i_str++) {

      if (id_thread == n_cores - 1) {
        const int64_t i_progress = i_iteration * length_str + i_str;
        if (i_progress % SIZE_CHUNK_PROGRESSBAR == 0) {
          // Print progress
          const double percent = 100 * (double)i_progress / (n_iteration * length_str);
          std::wcout << "\rProgress : "
                     << std::fixed << std::setprecision(2) << percent
                     << "%     " << std::flush;
        }
      }

      double ratio_completed = (i_iteration*length_str + i_str) / static_cast<double>(n_iteration*length_str + 1);
      if (ratio_completed > 0.9999) ratio_completed = 0.9999;
      const double _learning_rate = learning_rate * (1 - ratio_completed);

      // For each (center) word for every n-gram
      for (int64_t length_word=1; length_word<=max_length_word; length_word++) {
        if (i_str + length_word - 1 >= length_str) break;

        const std::wstring word = corpus_thread.substr(i_str, length_word);
        if (vocabulary2id_thread.find(word) == vocabulary2id_thread.end()) continue;
        const int64_t id_word = vocabulary2id_thread[word];

        const int64_t freq = count_vocabulary_thread[id_word];
        const double probability_reject = (sqrt(freq/(rate_sample*sum_count_vocabulary)) + 1) * (rate_sample*sum_count_vocabulary) / freq;
        if (probability_reject < cheaprand_thread.generate_rand_uniform(0, 1)) continue;

        // For each context word
        for (int64_t length_context=1; length_context<=max_length_word; length_context++) {
          if (i_str + length_word + length_context - 1 >= length_str) break;

          const std::wstring context = corpus_thread.substr(i_str + length_word, length_context);
          if (vocabulary2id_thread.find(context) == vocabulary2id_thread.end()) continue;
          const int64_t id_context = vocabulary2id_thread[context];

          //// Skip-gram with negative sampling

          // Vector representation of `word` can be obtained by
          //  (embeddings_words[i_head_word], ..., embeddings_words[i_head_word + dim_embedding - 1]).
          int64_t i_head_word, i_head_context, i_head_target;

          for (const bool is_right_context : {true, false}) {

            if (is_right_context) { // Right context
              i_head_word = dim_embedding * id_word;
              i_head_context = dim_embedding * id_context;
            } else { // Left context
              i_head_word = dim_embedding * id_context;
              i_head_context = dim_embedding * id_word;
            }

            for (int64_t i=0; i<dim_embedding; i++) {
              gradient_words[i] = 0;
            }

            for (int64_t i_ns=-1; i_ns<n_negative_sample; i_ns++) {
              const bool is_negative_sample = (i_ns >= 0);

              if (is_negative_sample) {
                i_head_target = dim_embedding * table_unigram[cheaprand_thread.generate_randint(SIZE_TABLE_UNIGRAM)];
                if (i_head_target == i_head_context) {
                  continue;
                }
              } else {
                i_head_target = i_head_context;
              }

              double x = 0; // inner product
              for (int64_t i=0; i<dim_embedding; i++) {
                if (is_right_context) {
                  x += embeddings_words[i_head_word + i] * embeddings_contexts_right[i_head_target + i];
                } else {
                  x += embeddings_words[i_head_word + i] * embeddings_contexts_left[i_head_target + i];
                }
              }

              const double g = 1. / (1. + exp(-x)) - (1.0 - (double)is_negative_sample);
              for (int64_t i=0; i<dim_embedding; i++) {
                if (is_right_context) {
                  gradient_words[i] += g * embeddings_contexts_right[i_head_target + i];
                  embeddings_contexts_right[i_head_target + i] -= _learning_rate * g * embeddings_words[i_head_word + i];
                } else {
                  gradient_words[i] += g * embeddings_contexts_left[i_head_target + i];
                  embeddings_contexts_left[i_head_target + i] -= _learning_rate * g * embeddings_words[i_head_word + i];
                }
              }

            }

            for (int64_t i=0; i<dim_embedding; i++) {
              embeddings_words[i_head_word + i] -= _learning_rate * gradient_words[i];
            }

          }
        }
      }
    }
  }

  delete[] gradient_words;
  if (id_thread == n_cores - 1) std::wcout << std::endl << std::flush;
}

void SkipGram::construct_unigramtable(const double power_unigram_table) {
  table_unigram = new int64_t[SIZE_TABLE_UNIGRAM];
  double sum_count_power = 0;

  for (auto c : count_vocabulary) {
    sum_count_power += pow(c, power_unigram_table);
  }

  int64_t id_word = 0;
  double cumsum_count_power = pow(count_vocabulary[id_word], power_unigram_table)/sum_count_power;

  for (int64_t i_table=0; i_table<SIZE_TABLE_UNIGRAM; i_table++) {
    table_unigram[i_table] = id_word;
    if (i_table / static_cast<double>(SIZE_TABLE_UNIGRAM) > cumsum_count_power) {
      id_word++;
      cumsum_count_power += pow(count_vocabulary[id_word], power_unigram_table)/sum_count_power;
    }
    if (id_word >= size_vocabulary) id_word = size_vocabulary - 1;
  }
}

void SkipGram::save_vector(const std::string output_path)
{
  std::cout << "Saving embeddings to " << output_path << std::endl;

  std::wofstream fout(output_path);
  fout << size_vocabulary << " " << dim_embedding << std::endl;
  for (int64_t i=0; i<size_vocabulary; i++) {
    fout << vocabulary[i] << " ";
    for (int64_t j=0; j<dim_embedding; j++) {
      fout << embeddings_words[i*dim_embedding + j];
      if (j < dim_embedding - 1) {
        fout << " ";
      }else{
        fout << std::endl;
      }
    }
  }
  fout.close();

  std::cout << "Done" << std::endl;
}
