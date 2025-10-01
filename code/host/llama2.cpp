/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <algorithm>
#include "typedefs.h"
#include "forward.h"
#include "config.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#include <numeric>
#endif

/*-----------------------------------------------------------------------------------------*/
// Data structures

// Structure to hold token and its index in the vocabulary
typedef struct
{
  char *str;
  int id;
} TokenIndex;

// Structure to hold the tokenizer data
typedef struct
{
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;

// Structure to hold quantized tensor (int8 values + scales)
typedef struct
{
  float prob;
  int index;
} ProbIndex; 

// Sampler state 
typedef struct
{
  int vocab_size;
  ProbIndex *probindex;
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

// Benchmarking and evaluation metrics
typedef struct {
  // Perplexity metrics
  float total_log_prob;
  int total_tokens;
  int vocab_size;
  bool enabled;
  
  // Timing metrics
  long total_inference_time_ms;
  long total_first_token_time_ms;
  int total_stories;
  int total_inference_calls;
  
  // Per-story statistics for analysis
  std::vector<float> story_throughputs;
  std::vector<long> story_first_token_latencies;
  std::vector<int> story_token_counts;
} BenchmarkEval;


/*-----------------------------------------------------------------------------------------*/
// Utilities: Sampling

// Applies softmax to an array of floats in place
void softmax(float *x, int size) {
  // Find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  // Exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  // Normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// Greedy argmax sampling
int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

// Multinomial sampling from the probability distribution
int sample_mult(float *probabilities, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

// Helper function for qsort, sorts ProbIndex in descending order of prob
int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

// Top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability topp. 
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  int n0 = 0;

  // Values smaller than (1 - topp) / (n - 1) are removed for efficiency
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }

  // Quicksort indices in descending order of probabilities
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1;

  // Truncate the list where cumulative probability exceeds topp
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // Exceeded topp by including last_idx
    }
  }

  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  // Sample from the truncated list
  for (int i = 0; i <= last_idx; i++)
  {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // In case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));   // buffer only used with nucleus sampling; may not need but it's ~small
}

// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
unsigned int random_u32(unsigned long long *state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// Random f32 (0,1]
float random_f32(unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}



/*-----------------------------------------------------------------------------------------*/
// Helpers: Model loading

// Dequantizes a list of quantized tensors into float tensors
template <int SIZE>
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each) {
  void *p = *ptr;
  for (int i = 0; i < n; i++) {
    std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
    p = (int8_t *)p + size_each;
    std::memcpy(tensor[i].s, p, (size_each / GS) * sizeof(float));
    p = (float *)p + size_each / GS;
  }
  *ptr = p;
}

// Maps weights from a contiguous memory buffer to the transformer weights structure
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier){
  int head_size = dim / n_heads;
  float *fptr = (float *)ptr;
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  ptr = (void *)fptr;
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  if (shared_classifier) {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  } else {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
  }
}

// Reads checkpoint file, validates header, and memory maps weights
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights)
{
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }

  // Validate magic number ("ak42") and version (2)
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 || magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1 || version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }

  int header_size = 256; // header size for version 2

  // Read model config (excluding GS)
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1) {
    fprintf(stderr, "Failed to read config\n");
    exit(EXIT_FAILURE);
  }

  // Read flags
  uint8_t shared_classifier;
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
    fprintf(stderr, "Failed to read shared_classifier flag\n");
    exit(EXIT_FAILURE);
  }
  int group_size;
  if (fread(&group_size, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "Failed to read group_size\n");
    exit(EXIT_FAILURE);
  }
  config->GS = GS;

  // Get file size and close file
  fseek(file, 0, SEEK_END);
  auto file_size = ftell(file);
  fclose(file);

  // Memory map weights
  int fd = open(checkpoint.c_str(), O_RDONLY);
  if (fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    close(fd);
    exit(EXIT_FAILURE);
  }
  void *weights_ptr = ((char *)data) + header_size;
  memory_map_weights(weights, weights_ptr, shared_classifier);

  close(fd);
  munmap(data, file_size);
}



/*-----------------------------------------------------------------------------------------*/
// Helpers: Encoding & Decoding

// Helper function for qsort, sorts TokenIndex in ascending order of str
int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

// Efficiently find the perfect match for str in vocab, return its index or -1 if not found
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  TokenIndex tok = {.str = str}; // Acts as the key to search for
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}



/*-----------------------------------------------------------------------------------------*/
// Model & Tokenizer Builders

// Read in the Config and the Weights from the checkpoint
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string checkpoint_path) {
  read_checkpoint(checkpoint_path, &t->config, &t->weights);
}

// Build the tokenizer from the tokenizer file
void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL;

  // Initialize the byte pieces
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }

  // Read in the file
  FILE *file = fopen(tokenizer_path.c_str(), "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str());
    exit(EXIT_FAILURE);
  }

  // Read in the max token length (uint32)
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }

  int len;
  // Read in each token's score and string
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // Add string terminating token
  }
  fclose(file);
}



/*-----------------------------------------------------------------------------------------*/
// Utilities: Tokenization & Sampling

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  // Following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') piece++;

  unsigned char byte_val;

  // If this is a raw byte token, map it to the corresponding byte piece
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

// BPE encode function
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  // Create the sorted vocabulary
  if (t->sorted_vocab == NULL) {
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // Buffer to hold the current UTF-8 codepoint being processed
  char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 1;

  // Add a dummy space prefix if the input text is not empty
  if (text[0] != '\0') {
    const char* dummy_prefix_str = " ";
    int dummy_prefix = str_lookup((char*)dummy_prefix_str, t->sorted_vocab, t->vocab_size);
    if (dummy_prefix != -1) {
      tokens[(*n_tokens)++] = dummy_prefix;
    }
  }

  // Process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // If this byte is not a UTF-8 continuation byte (i.e. does not start with 10xxxxxx)
    if ((*c & 0xC0) != 0x80) str_len = 0;
  
    // Append the current byte to the buffer
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';

    // If the next byte is a continuation byte
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size); // look for this codepoint in vocab

    if (id != -1) {
      tokens[(*n_tokens)++] = id; // Find codepoint in vocab, add it as a token
    } else {
      // Fallback: encode each byte as a token (offset by 3 for special tokens)
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // Protect against a sequence of stray UTF8 continuation bytes
  }

  // Merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    // Check if we can merge the pair (tokens[i], tokens[i+1])
    for (int i = 0; i < (*n_tokens - 1); i++) { 
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      // Check if this merge pair exists in vocab and if it has the best score so far
      if (id != -1 && t->vocab_scores[id] > best_score) {
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) break; // No more pairs can be merged
    tokens[best_idx] = best_id; // Replace token at position best_idx with the merged token best_id

    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // Token length decreased
  }

  // Add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// Sample the token given the logits and some hyperparameters
int sample(Sampler *sampler, float *logits) {
  int next;
  if (sampler->temperature == 0.0f) {  // Greedy argmax sampling
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // Apply the temperature
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;  
    }

    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state); // Source of entropy for sampling

    if (sampler->topp <= 0 || sampler->topp >= 1) {
      next = sample_mult(logits, sampler->vocab_size, coin); // Simply sample from the predicted probability distribution
    }
    else {
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin); // Top-p (nucleus) sampling, clamping the least likely tokens to zero
    }
  }
  return next;
}

/*-----------------------------------------------------------------------------------------*/
// Utilities: Generation & Evaluation

// Safe print function for decoded pieces
void safe_printf(char *piece) {
  if (piece == NULL) return;
  if (piece[0] == '\0') return;
  
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    // Bad byte, don't print it
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; 
    }
  }
  printf("%s", piece);
}

long time_in_ms() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Zero out  cache buffers
void clear_kv_cache(xrt::bo& key_buffer, xrt::bo& value_buffer, int cache_dim) {
    std::vector<float> zero_cache(cache_dim, 0.0f);
    key_buffer.write(zero_cache.data(), cache_dim * sizeof(float), 0);
    value_buffer.write(zero_cache.data(), cache_dim * sizeof(float), 0);
    key_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    value_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

// Compute log probability of the target token given the logits
float compute_log_prob(float *logits, int target_token, int vocab_size) {
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute softmax denominator
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    // Calculate log probability
    float target_logit = logits[target_token];
    float log_prob = (target_logit - max_logit) - logf(sum_exp);
    
    return log_prob;
}

// Update perplexity evaluation metrics based on the logits and target token
void update_perplexity_eval(BenchmarkEval* eval, float* logits, int target_token) {
    if (!eval || !eval->enabled) return;
    
    float log_prob = compute_log_prob(logits, target_token, eval->vocab_size);
    eval->total_log_prob += log_prob;
    eval->total_tokens++;
}

// Calculate final perplexity from accumulated log probabilities and token count
float calculate_final_perplexity(const BenchmarkEval* eval) {
    if (!eval || eval->total_tokens == 0) return -1.0f;
    
    float avg_nll = -eval->total_log_prob / eval->total_tokens;
    return expf(avg_nll);
}

// Print comprehensive evaluation results
void print_comprehensive_results(BenchmarkEval& eval, long total_wall_time) {
    printf("\n=== EVALUATION RESULTS ===\n");
    
    // Perplexity Results
    float perplexity = calculate_final_perplexity(&eval);
    printf("PERPLEXITY METRICS:\n");
    printf("  Final perplexity: %.6f\n", perplexity);
    printf("  Total tokens evaluated: %d\n", eval.total_tokens);
    printf("  Average negative log likelihood: %.6f\n", -eval.total_log_prob / eval.total_tokens);
    
    // Throughput Results
    printf("\nTHROUGHPUT METRICS:\n");
    printf("  Stories processed: %d\n", eval.total_stories);
    printf("  Total inference calls: %d\n", eval.total_inference_calls);
    printf("  Total inference time: %ld ms\n", eval.total_inference_time_ms);
    printf("  Pure inference throughput: %.2f tok/s\n", 
           eval.total_tokens / (float)eval.total_inference_time_ms * 1000.0f);
    
    // First Token Latency
    printf("\nLATENCY METRICS:\n");
    printf("  Average first token latency: %.2f ms\n", 
           eval.total_first_token_time_ms / (float)eval.total_stories);
    
    // Statistical Analysis
    if (!eval.story_throughputs.empty()) {
        auto throughputs = eval.story_throughputs;
        std::sort(throughputs.begin(), throughputs.end());
        
        printf("\nSTATISTICAL ANALYSIS:\n");
        printf("  Throughput - Min: %.2f, Median: %.2f, Max: %.2f tok/s\n",
               throughputs.front(), 
               throughputs[throughputs.size()/2], 
               throughputs.back());
        
        // Calculate standard deviation
        float mean = std::accumulate(throughputs.begin(), throughputs.end(), 0.0f) / throughputs.size();
        float variance = 0.0f;
        for (float t : throughputs) {
            variance += (t - mean) * (t - mean);
        }
        float stddev = sqrt(variance / throughputs.size());
        printf("  Throughput - Mean: %.2f ± %.2f tok/s\n", mean, stddev);
    }
    
    printf("\nOVERALL TIMING:\n");
    printf("  Total wall time: %ld ms\n", total_wall_time);
    printf("  Effective utilization: %.1f%%\n", 
           100.0f * eval.total_inference_time_ms / total_wall_time);
}

// Process a single story: tokenization, inference loop, and metric updates
template<int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void process_story(const std::string& story_text, int story_num, Tokenizer* tokenizer, Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* transformer, xrt::kernel& kernel, xrt::bo& transformer_buffer, xrt::bo& key_buffer, xrt::bo& value_buffer, xrt::bo& out_buffer, float* logits, BenchmarkEval& eval) {
    
    if (story_text.empty()) return;

    // Tokenize story
    int num_tokens = 0;
    int* tokens = (int*)malloc((story_text.length() + 3) * sizeof(int));
    char* mutable_text = strdup(story_text.c_str());
    encode(tokenizer, mutable_text, 1, 0, tokens, &num_tokens);
    free(mutable_text);

    if (num_tokens < 2) {
        free(tokens);
        return;
    }

    printf("Story %d: %d tokens\n", story_num, num_tokens);

    // Timing variables
    long story_start_time = time_in_ms();
    long first_token_time = 0;
    long total_story_inference_time = 0;
    int inference_calls = 0;

    // Process each token
    for (int local_pos = 0; local_pos < num_tokens - 1; local_pos++) {
        int current_token = tokens[local_pos];
        int target_token = tokens[local_pos + 1];

        // Time individual inference call
        long inference_start = time_in_ms();
        
        auto run = kernel(transformer_buffer, current_token, local_pos,
                         key_buffer, value_buffer, out_buffer);
        run.wait();
        
        long inference_end = time_in_ms();
        long inference_time = inference_end - inference_start;
        
        // Track first token latency separately
        if (local_pos == 0) first_token_time = inference_time;
        
        total_story_inference_time += inference_time;
        inference_calls++;

        // Read results and update perplexity
        out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        out_buffer.read(logits, transformer->config.vocab_size * sizeof(float), 0);
        
        update_perplexity_eval(&eval, logits, target_token);
    }

    // Calculate story-level metrics
    float story_throughput = (num_tokens - 1) / (float)total_story_inference_time * 1000.0f;
    
    // Update benchmark statistics
    eval.total_inference_time_ms += total_story_inference_time;
    eval.total_first_token_time_ms += first_token_time;
    eval.total_stories++;
    eval.total_inference_calls += inference_calls;
    
    eval.story_throughputs.push_back(story_throughput);
    eval.story_first_token_latencies.push_back(first_token_time);
    eval.story_token_counts.push_back(num_tokens - 1);

    printf("  Story %d: %.2f tok/s, first token: %ld ms\n", 
           story_num, story_throughput, first_token_time);

    free(tokens);
}



/*-----------------------------------------------------------------------------------------*/
// Generation Loop

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void generate(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, std::string &kernelpath)
{
  const char *empty_prompt = "";
  if (prompt == NULL)
  {
    prompt = (char*)empty_prompt;
  }

  // Encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1)
  {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading kernel..." << std::endl;
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(kernelpath);
  auto kernel = xrt::kernel(device, uuid, "forward");
  std::cout << "Out buffer size: " << vocab_size * sizeof(float) << std::endl;
  std::cout << "Transformer size: " << sizeof(*transformer) << std::endl;
  std::cout << "Allocating output buffer" << std::endl;
  auto out_buffer = xrt::bo(device, vocab_size * sizeof(float), kernel.group_id(5));

  int cache_dim = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
  std::cout << "Allocating buffers" << std::endl;
  auto transformer_buffer = xrt::bo(device, sizeof(*transformer), kernel.group_id(0));

  auto key_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(3));
  auto value_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(4));

  std::cout << "Copying data to buffer" << std::endl;
  transformer_buffer.write(transformer, sizeof(*transformer), 0);

  transformer_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  // First run: process the initial token
  auto run = kernel(transformer_buffer, token, pos, key_buffer, value_buffer, out_buffer);
  run.wait();

  float *logits = (float *)malloc(vocab_size * sizeof(float));
  out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  out_buffer.read(logits, vocab_size * sizeof(float), 0);

  // Advance the state machine: use prompt tokens if available, otherwise sample.
  if (pos < num_prompt_tokens - 1) {
    next = prompt_tokens[pos + 1];
  } else {
    next = sample(sampler, logits);
  }
  pos++;

  // Print the token as string, decode it with the Tokenizer object
  char *piece = decode(tokenizer, token, next);
  safe_printf(piece); // printf("%s", piece), but skips "unsafe" bytes
  fflush(stdout);
  token = next;
  start = time_in_ms();

  // Main generation loop
  while (pos < steps) {
    run.set_arg(1, token);
    run.set_arg(2, pos);
    run.start();
    run.wait();
    out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_buffer.read(logits, vocab_size * sizeof(float), 0);

    // Advance state machine: use prompt tokens if available, otherwise sample.
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      next = sample(sampler, logits);
    }
    pos++;

    // Stop generation if BOS (=1) token is produced.
    if (next == 1) break;

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "Total (prompt & generation) achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

/*-----------------------------------------------------------------------------------------*/
// Evaluation Loop

template<int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void evaluate_with_benchmarking(std::string text_file, 
                               Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* transformer, 
                               Tokenizer* tokenizer, 
                               const std::string& kernelpath,
                               int max_stories = 100) {

    printf("=== EVALUATION (BENCH/PERPLEXITY) ===\n");
    printf("File: %s\n", text_file.c_str());
    printf("Max stories: %s\n", max_stories == -1 ? "unlimited" : std::to_string(max_stories).c_str());
    
    FILE* file = fopen(text_file.c_str(), "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open %s\n", text_file.c_str());
        return;
    }
    
    BenchmarkEval eval = {
        .total_log_prob = 0.0f,
        .total_tokens = 0,
        .vocab_size = vocab_size,
        .enabled = true,
        .total_inference_time_ms = 0,
        .total_first_token_time_ms = 0,
        .total_stories = 0,
        .total_inference_calls = 0
    };
    
    // FPGA setup
    printf("Initializing FPGA kernel...\n");
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(kernelpath);
    auto kernel = xrt::kernel(device, uuid, "forward");
    
    auto out_buffer = xrt::bo(device, vocab_size * sizeof(float), kernel.group_id(5));
    int cache_dim = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
    
    auto transformer_buffer = xrt::bo(device, sizeof(*transformer), kernel.group_id(0));
    auto key_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(3));
    auto value_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(4));
    
    transformer_buffer.write(transformer, sizeof(*transformer), 0);
    transformer_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    float* logits = (float*)malloc(transformer->config.vocab_size * sizeof(float));

    // Story processing
    std::string current_story = "";
    char line[4096];
    int stories_processed = 0;
    
    printf("Starting evaluation...\n");
    long overall_start = time_in_ms();
    
    // Main processing loop
    while (fgets(line, sizeof(line), file)) {
        if (max_stories > 0 && stories_processed >= max_stories) break;

        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 0) continue;
        
        if (strcmp(line, "<|endoftext|>") == 0) {
            if (!current_story.empty()) {
                // Clear cache before each story for consistent measurements
                clear_kv_cache(key_buffer, value_buffer, cache_dim);
                process_story<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(current_story, stories_processed + 1, tokenizer, transformer, kernel, transformer_buffer, key_buffer, value_buffer, out_buffer, logits, eval);
                stories_processed++;
                current_story = "";
            }
        } else {
            if (!current_story.empty()) current_story += " ";
            current_story += line;
        }
    }
    // Process final story
    if (!current_story.empty() && (max_stories == -1 || stories_processed < max_stories)) {
        clear_kv_cache(key_buffer, value_buffer, cache_dim);
        process_story<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(current_story, stories_processed + 1, tokenizer, transformer, kernel, transformer_buffer, key_buffer, value_buffer, out_buffer, logits, eval);        stories_processed++;
    }
    
    long overall_end = time_in_ms();
    fclose(file);
    free(logits);
    print_comprehensive_results(eval, overall_end - overall_start);
}

/*-----------------------------------------------------------------------------------------*/
// CLI
#ifndef TESTING

void error_usage()
{
  fprintf(stderr, "Usage:   ./llama2 <checkpoint> [options]\n");
  fprintf(stderr, "Example: ./llama2 model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Example: ./llama2 model.bin -m evaluate -e evalfile.txt\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat|evaluate, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -e <file>   evaluation input text file\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
  // Default parameters
  std::string checkpoint_path;
  std::string tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  char *prompt = nullptr;
  unsigned long long rng_seed = 0;
  std::string mode = "generate";
  char *system_prompt = nullptr;
  std::string kernelpath = "forward.xclbin";
  std::string eval_file = "TinyStoriesV2-GPT4-valid.txt";

  // Parse command line arguments
  if (argc < 2) {
    error_usage();
  }
  checkpoint_path = argv[1]; // First argument is the checkpoint/model file
  for (int i = 2; i < argc; i += 2) {
    // Each option should be in the form -x <value>
    if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) {
      error_usage();
    }
    switch (argv[i][1]) {
      case 't': // Temperature for sampling
        temperature = atof(argv[i + 1]);
        break;
      case 'p': // Top-p (nucleus) sampling value
        topp = atof(argv[i + 1]);
        break;
      case 's': // Random seed
        rng_seed = atoi(argv[i + 1]);
        break;
      case 'n': // Number of generation steps
        steps = atoi(argv[i + 1]);
        break;
      case 'i': // Input prompt string
        prompt = argv[i + 1];
        break;
      case 'z': // Custom tokenizer path
        tokenizer_path = argv[i + 1];
        break;
      case 'm': // Mode (generate, chat, etc.)
        mode = argv[i + 1];
        break;
      case 'y': // System prompt for chat mode
        system_prompt = argv[i + 1];
        break;
      case 'k': // Kernel binary path
        kernelpath = argv[i + 1];
        break;
      case 'e': { // Evaluate perplexity on a text file
        eval_file = argv[i + 1];
      }
      default:
        error_usage(); // Unknown option
    }
  }

  // Parameter validation
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0f) temperature = 0.0f;
  if (topp < 0.0f || topp > 1.0f) topp = 0.9f;

  // Build Transformer
  static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps <= 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len;

  // Build Tokenizer
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // Build Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // Run generation or other modes
  if (mode == "generate") {
    generate(&transformer, &tokenizer, &sampler, prompt, steps, kernelpath);
  } else if (mode == "evaluate") {
    evaluate_with_benchmarking<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(eval_file, &transformer, &tokenizer, kernelpath);  } else {
    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    error_usage();
  }

  // Cleanup
  free(sampler.probindex);
  for (int i = 0; i < tokenizer.vocab_size; i++) free(tokenizer.vocab[i]);
  free(tokenizer.vocab);
  free(tokenizer.vocab_scores);
  free(tokenizer.sorted_vocab);
  return 0;
}
#endif
