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

struct Weights {
    // Embedding (dequantized for host use)
    float *token_embedding_table;
    
    // Attention weights (quantized - keep in int8)
    int8_t *wq_weights;
    float *wq_scales;
    int8_t *wk_weights;
    float *wk_scales;
    int8_t *wv_weights;
    float *wv_scales;
    int8_t *wo_weights;
    float *wo_scales;
    
    // FFN weights (quantized)
    int8_t *w1_weights;
    float *w1_scales;
    int8_t *w2_weights;
    float *w2_scales;
    int8_t *w3_weights;
    float *w3_scales;
    
    // RMS norm weights (float)
    float *rms_att_weight;
    float *rms_ffn_weight;
    float *rms_final_weight;
    
    // Classifier weights (quantized)
    int8_t *wcls_weights;
    float *wcls_scales;
};


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

// Bundle for device BOs
struct DeviceBOs {
    xrt::bo emb_bo;
    xrt::bo wq_bo;   xrt::bo wq_s_bo;
    xrt::bo wk_bo;   xrt::bo wk_s_bo;
    xrt::bo wv_bo;   xrt::bo wv_s_bo;
    xrt::bo wo_bo;   xrt::bo wo_s_bo;
    xrt::bo w1_bo;   xrt::bo w1_s_bo;
    xrt::bo w2_bo;   xrt::bo w2_s_bo;
    xrt::bo w3_bo;   xrt::bo w3_s_bo;
    xrt::bo rms_att_bo;
    xrt::bo rms_ffn_bo;
    xrt::bo rms_final_bo;
    xrt::bo wcls_bo; xrt::bo wcls_s_bo;
    xrt::bo key_bo;  xrt::bo value_bo;
    xrt::bo out_bo;

    DeviceBOs(xrt::device& dev, xrt::kernel& k,
              size_t emb_bytes, size_t att_size, size_t att_scale_bytes,
              size_t ffn1_size, size_t ffn1_scale_bytes,
              size_t ffn2_size, size_t ffn2_scale_bytes,
              size_t cls_size, size_t cls_scale_bytes, size_t cache_dim, size_t out_bytes)
      : emb_bo(dev, emb_bytes, k.group_id(0)),
        wq_bo(dev, att_size, k.group_id(1)), wq_s_bo(dev, att_scale_bytes, k.group_id(2)),
        wk_bo(dev, att_size, k.group_id(3)), wk_s_bo(dev, att_scale_bytes, k.group_id(4)),
        wv_bo(dev, att_size, k.group_id(5)), wv_s_bo(dev, att_scale_bytes, k.group_id(6)),
        wo_bo(dev, att_size, k.group_id(7)), wo_s_bo(dev, att_scale_bytes, k.group_id(8)),
        w1_bo(dev, ffn1_size, k.group_id(9)), w1_s_bo(dev, ffn1_scale_bytes, k.group_id(10)),
        w2_bo(dev, ffn2_size, k.group_id(11)), w2_s_bo(dev, ffn2_scale_bytes, k.group_id(12)),
        w3_bo(dev, ffn1_size, k.group_id(13)), w3_s_bo(dev, ffn1_scale_bytes, k.group_id(14)),
        rms_att_bo(dev, n_layers * dim * sizeof(float), k.group_id(15)),
        rms_ffn_bo(dev, n_layers * dim * sizeof(float), k.group_id(16)),
        rms_final_bo(dev, dim * sizeof(float), k.group_id(17)),
        wcls_bo(dev, cls_size, k.group_id(18)), wcls_s_bo(dev, cls_scale_bytes, k.group_id(19)),
        key_bo(dev, cache_dim * sizeof(float), k.group_id(20)),
        value_bo(dev, cache_dim * sizeof(float), k.group_id(21)),
        out_bo(dev, out_bytes, k.group_id(22))
    {}
};

// Allocate BOs, upload weights, zero caches
DeviceBOs prepare_device_bos(xrt::device& device, xrt::kernel& kernel, Weights* weights) {
    // compute sizes (must match forward.cpp assumptions)
    size_t emb_bytes = (size_t)vocab_size * dim * sizeof(float);
    size_t att_size = (size_t)n_layers * dim * dim;
    size_t att_scale_bytes = (att_size / GS) * sizeof(float);
    size_t ffn1_size = (size_t)n_layers * dim * hidden_dim;
    size_t ffn1_scale_bytes = (ffn1_size / GS) * sizeof(float);
    size_t ffn2_size = (size_t)n_layers * hidden_dim * dim;
    size_t ffn2_scale_bytes = (ffn2_size / GS) * sizeof(float);
    size_t cls_size = (size_t)vocab_size * dim;
    size_t cls_scale_bytes = (cls_size / GS) * sizeof(float);
    size_t cache_dim = (size_t)n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
    size_t out_bytes = (size_t)vocab_size * sizeof(float);

    DeviceBOs bos(device, kernel,
                  emb_bytes, att_size, att_scale_bytes,
                  ffn1_size, ffn1_scale_bytes,
                  ffn2_size, ffn2_scale_bytes,
                  cls_size, cls_scale_bytes, cache_dim, out_bytes);

    // write weights (order must match kernel arg mapping)
    bos.emb_bo.write(weights->token_embedding_table, emb_bytes, 0); bos.emb_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wq_bo.write(weights->wq_weights, att_size, 0); bos.wq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wq_s_bo.write(weights->wq_scales, att_scale_bytes, 0); bos.wq_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wk_bo.write(weights->wk_weights, att_size, 0); bos.wk_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wk_s_bo.write(weights->wk_scales, att_scale_bytes, 0); bos.wk_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wv_bo.write(weights->wv_weights, att_size, 0); bos.wv_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wv_s_bo.write(weights->wv_scales, att_scale_bytes, 0); bos.wv_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wo_bo.write(weights->wo_weights, att_size, 0); bos.wo_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wo_s_bo.write(weights->wo_scales, att_scale_bytes, 0); bos.wo_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w1_bo.write(weights->w1_weights, ffn1_size, 0); bos.w1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w1_s_bo.write(weights->w1_scales, ffn1_scale_bytes, 0); bos.w1_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w2_bo.write(weights->w2_weights, ffn2_size, 0); bos.w2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w2_s_bo.write(weights->w2_scales, ffn2_scale_bytes, 0); bos.w2_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w3_bo.write(weights->w3_weights, ffn1_size, 0); bos.w3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.w3_s_bo.write(weights->w3_scales, ffn1_scale_bytes, 0); bos.w3_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.rms_att_bo.write(weights->rms_att_weight, n_layers * dim * sizeof(float), 0); bos.rms_att_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.rms_ffn_bo.write(weights->rms_ffn_weight, n_layers * dim * sizeof(float), 0); bos.rms_ffn_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.rms_final_bo.write(weights->rms_final_weight, dim * sizeof(float), 0); bos.rms_final_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wcls_bo.write(weights->wcls_weights, cls_size, 0); bos.wcls_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.wcls_s_bo.write(weights->wcls_scales, cls_scale_bytes, 0); bos.wcls_s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // zero key/value cache
    std::vector<float> zero_cache(cache_dim, 0.0f);
    bos.key_bo.write(zero_cache.data(), cache_dim * sizeof(float), 0); bos.key_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bos.value_bo.write(zero_cache.data(), cache_dim * sizeof(float), 0); bos.value_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    return bos;
}


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


void read_checkpoint(std::string checkpoint, Weights *weights) {
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open %s\n", checkpoint.c_str()); 
        exit(1); 
    }
    
    // Read and validate header
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) { 
        fprintf(stderr, "Bad magic number\n"); 
        exit(1); 
    }
    
    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 2) { 
        fprintf(stderr, "Unsupported version %d\n", version); 
        exit(1); 
    }
    
    // Read config (skip, already have it from config.h)
    Config temp_config;
    int header_size = 256;
    fread(&temp_config, sizeof(Config) - sizeof(int), 1, file);
    
    uint8_t shared_classifier;
    fread(&shared_classifier, sizeof(uint8_t), 1, file);
    
    int group_size;
    fread(&group_size, sizeof(int), 1, file);
    
    // Seek to start of weights
    fseek(file, header_size, SEEK_SET);
    
    // Calculate sizes
    const int kv_dim = (dim * n_kv_heads) / n_heads;
    
    // Allocate all weight arrays
    weights->rms_att_weight = (float *)malloc(n_layers * dim * sizeof(float));
    weights->rms_ffn_weight = (float *)malloc(n_layers * dim * sizeof(float));
    weights->rms_final_weight = (float *)malloc(dim * sizeof(float));
    
    // Read RMS norm weights
    fread(weights->rms_att_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_ffn_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_final_weight, sizeof(float), dim, file);
    
    // Token embedding - read quantized, then dequantize
    int emb_size = vocab_size * dim;
    int8_t *q_emb = (int8_t *)malloc(emb_size * sizeof(int8_t));
    float *s_emb = (float *)malloc((emb_size / GS) * sizeof(float));
    
    fread(q_emb, sizeof(int8_t), emb_size, file);
    fread(s_emb, sizeof(float), emb_size / GS, file);
    
    // Dequantize embedding for host
    weights->token_embedding_table = (float *)malloc(emb_size * sizeof(float));
    for (int i = 0; i < emb_size; i++) {
        weights->token_embedding_table[i] = q_emb[i] * s_emb[i / GS];
    }
    
    // Attention weights (all layers)
    int att_size = n_layers * dim * dim;
    int att_scale_size = att_size / GS;
    
    weights->wq_weights = (int8_t *)malloc(att_size * sizeof(int8_t));
    weights->wq_scales = (float *)malloc(att_scale_size * sizeof(float));
    weights->wk_weights = (int8_t *)malloc(att_size * sizeof(int8_t));
    weights->wk_scales = (float *)malloc(att_scale_size * sizeof(float));
    weights->wv_weights = (int8_t *)malloc(att_size * sizeof(int8_t));
    weights->wv_scales = (float *)malloc(att_scale_size * sizeof(float));
    weights->wo_weights = (int8_t *)malloc(att_size * sizeof(int8_t));
    weights->wo_scales = (float *)malloc(att_scale_size * sizeof(float));
    
    // Read attention weights
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wq_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wq_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wk_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wk_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wv_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wv_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * dim;
        fread(&weights->wo_weights[offset], sizeof(int8_t), dim * dim, file);
        fread(&weights->wo_scales[offset / GS], sizeof(float), dim * dim / GS, file);
    }
    
    // FFN weights (all layers)
    int ffn1_size = n_layers * dim * hidden_dim;
    int ffn1_scale_size = ffn1_size / GS;
    int ffn2_size = n_layers * hidden_dim * dim;
    int ffn2_scale_size = ffn2_size / GS;
    
    weights->w1_weights = (int8_t *)malloc(ffn1_size * sizeof(int8_t));
    weights->w1_scales = (float *)malloc(ffn1_scale_size * sizeof(float));
    weights->w2_weights = (int8_t *)malloc(ffn2_size * sizeof(int8_t));
    weights->w2_scales = (float *)malloc(ffn2_scale_size * sizeof(float));
    weights->w3_weights = (int8_t *)malloc(ffn1_size * sizeof(int8_t));
    weights->w3_scales = (float *)malloc(ffn1_scale_size * sizeof(float));
    
    // Read FFN weights
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * hidden_dim;
        fread(&weights->w1_weights[offset], sizeof(int8_t), dim * hidden_dim, file);
        fread(&weights->w1_scales[offset / GS], sizeof(float), dim * hidden_dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * hidden_dim * dim;
        fread(&weights->w2_weights[offset], sizeof(int8_t), hidden_dim * dim, file);
        fread(&weights->w2_scales[offset / GS], sizeof(float), hidden_dim * dim / GS, file);
    }
    for (int l = 0; l < n_layers; l++) {
        int offset = l * dim * hidden_dim;
        fread(&weights->w3_weights[offset], sizeof(int8_t), dim * hidden_dim, file);
        fread(&weights->w3_scales[offset / GS], sizeof(float), dim * hidden_dim / GS, file);
    }
    
    // Classifier weights
    int cls_size = vocab_size * dim;
    int cls_scale_size = cls_size / GS;
    
    if (!shared_classifier) {
        weights->wcls_weights = (int8_t *)malloc(cls_size * sizeof(int8_t));
        weights->wcls_scales = (float *)malloc(cls_scale_size * sizeof(float));
        
        fread(weights->wcls_weights, sizeof(int8_t), cls_size, file);
        fread(weights->wcls_scales, sizeof(float), cls_scale_size, file);
    } else {
        // Reuse token embedding weights
        weights->wcls_weights = q_emb;  // Keep quantized version
        weights->wcls_scales = s_emb;
    }
    
    fclose(file);
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

static void error_usage() {
    fprintf(stderr,
        "Usage: llama2_host <checkpoint> [-t <temperature>] [-p <top_p>] [-s <seed>] [-n <steps>]\n"
        "                     [-i <prompt>] [-z <tokenizer>] [-m <mode>] [-k <kernel.xclbin>] [-e <eval_file>]\n");
    exit(1);
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

void process_story_flat(const std::string& story_text,
                        int story_num, Tokenizer* tokenizer,
                        Weights *weights, xrt::kernel& kernel,
                        xrt::bo& emb_bo,
                        xrt::bo& wq_bo, xrt::bo& wq_s_bo,
                        xrt::bo& wk_bo, xrt::bo& wk_s_bo,
                        xrt::bo& wv_bo, xrt::bo& wv_s_bo,
                        xrt::bo& wo_bo, xrt::bo& wo_s_bo,
                        xrt::bo& w1_bo, xrt::bo& w1_s_bo,
                        xrt::bo& w2_bo, xrt::bo& w2_s_bo,
                        xrt::bo& w3_bo, xrt::bo& w3_s_bo,
                        xrt::bo& rms_att_bo, xrt::bo& rms_ffn_bo, xrt::bo& rms_final_bo,
                        xrt::bo& wcls_bo, xrt::bo& wcls_s_bo,
                        xrt::bo& key_buffer, xrt::bo& value_buffer,
                        xrt::bo& out_buffer,
                        float* logits,
                        BenchmarkEval& eval) {
    if (story_text.empty()) return;

    // Tokenize story
    int num_tokens = 0;
    int* tokens = (int*)malloc((story_text.length() + 3) * sizeof(int));
    char* mutable_text = strdup(story_text.c_str());
    encode(tokenizer, mutable_text, 1, 0, tokens, &num_tokens);
    free(mutable_text);
    if (num_tokens < 2) { free(tokens); return; }

    printf("Story %d: %d tokens\n", story_num, num_tokens);

    long story_start_time = time_in_ms();
    long first_token_time = 0;
    long total_story_inference_time = 0;
    int inference_calls = 0;

    // Per-token inference
    for (int local_pos = 0; local_pos < num_tokens - 1; ++local_pos) {
        int current_token = tokens[local_pos];
        int target_token = tokens[local_pos + 1];

        long inf_start = time_in_ms();

        auto run = kernel(
            emb_bo,
            wq_bo, wq_s_bo,
            wk_bo, wk_s_bo,
            wv_bo, wv_s_bo,
            wo_bo, wo_s_bo,
            w1_bo, w1_s_bo,
            w2_bo, w2_s_bo,
            w3_bo, w3_s_bo,
            rms_att_bo,
            rms_ffn_bo,
            rms_final_bo,
            wcls_bo, wcls_s_bo,
            key_buffer,
            value_buffer,
            out_buffer,
            current_token,
            local_pos
        );
        run.wait();

        long inf_end = time_in_ms();
        long inf_time = inf_end - inf_start;
        if (local_pos == 0) first_token_time = inf_time;
        total_story_inference_time += inf_time;
        inference_calls++;

        out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        out_buffer.read(logits, vocab_size * sizeof(float), 0);

        update_perplexity_eval(&eval, logits, target_token);
    }

    float story_throughput = (num_tokens - 1) / (float)total_story_inference_time * 1000.0f;
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

void generate(Weights *weights, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, std::string &kernelpath)
{
    #pragma HLS none
    const char *empty_prompt = "";
    if (prompt == NULL) prompt = (char*)empty_prompt;

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "expected >=1 prompt token\n"); exit(1); }

    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(kernelpath);
    auto kernel = xrt::kernel(device, uuid, "forward");

    // prepare device BOs (weights uploaded, caches zeroed)
    DeviceBOs bos = prepare_device_bos(device, kernel, weights);

    // First call: run kernel with flattened args
    int token = prompt_tokens[0];
    int pos = 0;
    auto run = kernel(
        bos.emb_bo,
        bos.wq_bo, bos.wq_s_bo,
        bos.wk_bo, bos.wk_s_bo,
        bos.wv_bo, bos.wv_s_bo,
        bos.wo_bo, bos.wo_s_bo,
        bos.w1_bo, bos.w1_s_bo,
        bos.w2_bo, bos.w2_s_bo,
        bos.w3_bo, bos.w3_s_bo,
        bos.rms_att_bo,
        bos.rms_ffn_bo,
        bos.rms_final_bo,
        bos.wcls_bo, bos.wcls_s_bo,
        bos.key_bo,
        bos.value_bo,
        bos.out_bo,
        token,
        pos
    );
    run.wait();

    // read logits from the kernel output BO (use bos.out_bo)
    std::vector<float> logits(vocab_size);
    bos.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bos.out_bo.read(logits.data(), vocab_size * sizeof(float), 0);

    // advance / sampling logic (same as before)
    int next;
    if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
    else {
        // greedy sample as fallback
        int max_idx = 0;
        for (int i = 1; i < vocab_size; ++i) if (logits[i] > logits[max_idx]) max_idx = i;
        next = max_idx;
    }
    pos++;
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);
    token = next;

    // subsequent loop: reuse BOs, set kernel args token/pos and start
    long start = time_in_ms();
    while (pos < steps) {
        auto run2 = kernel(
            bos.emb_bo,
            bos.wq_bo, bos.wq_s_bo,
            bos.wk_bo, bos.wk_s_bo,
            bos.wv_bo, bos.wv_s_bo,
            bos.wo_bo, bos.wo_s_bo,
            bos.w1_bo, bos.w1_s_bo,
            bos.w2_bo, bos.w2_s_bo,
            bos.w3_bo, bos.w3_s_bo,
            bos.rms_att_bo,
            bos.rms_ffn_bo,
            bos.rms_final_bo,
            bos.wcls_bo, bos.wcls_s_bo,
            bos.key_bo,
            bos.value_bo,
            bos.out_bo,
            token,
            pos
        );
        run2.wait();
        bos.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bos.out_bo.read(logits.data(), vocab_size * sizeof(float), 0);

        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
        else {
            int max_idx = 0;
            for (int i = 1; i < vocab_size; ++i) if (logits[i] > logits[max_idx]) max_idx = i;
            next = max_idx;
        }
        pos++;
        if (next == 1) break;
        char *piece2 = decode(tokenizer, token, next);
        safe_printf(piece2); fflush(stdout);
        token = next;
    }
    printf("\n");

    // cleanup BOs (RAII will free); free host prompt tokens
    free(prompt_tokens);
}

/*-----------------------------------------------------------------------------------------*/
// Evaluation Loop

void evaluate(const std::string& text_file,
                        Weights *weights,
                        Tokenizer *tokenizer,
                        const std::string &kernelpath,
                        int max_stories = 25) {
    // FPGA setup
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(kernelpath);
    auto kernel = xrt::kernel(device, uuid, "forward");

    DeviceBOs bos = prepare_device_bos(device, kernel, weights);

    // compute cache size & zero buffer used to clear KV between stories
    const size_t cache_dim = (size_t)n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
    std::vector<float> zero_cache(cache_dim, 0.0f);

    // open file and iterate stories
    FILE *file = fopen(text_file.c_str(), "r");
    if (!file) {
        fprintf(stderr, "Error: cannot open %s\n", text_file.c_str());
        return;
    }

    // Running evaluation accumulators
    BenchmarkEval eval = {};
    eval.enabled = true;
    eval.vocab_size = vocab_size;
    eval.total_log_prob = 0.0f;
    eval.total_tokens = 0;
    eval.total_inference_time_ms = 0;
    eval.total_first_token_time_ms = 0;
    eval.total_stories = 0;
    eval.total_inference_calls = 0;

    int stories_processed = 0;
    char line[4096];
    std::string current_story;
    long overall_start = time_in_ms();

    // buffers used per-inference
    std::vector<float> logits(vocab_size);
    auto first_words = [](const std::string &s, int max_words, int max_chars) {
        std::string out;
        int words = 0;
        for (size_t i = 0; i < s.size() && (int)out.size() < max_chars; ++i) {
            char c = s[i];
            out.push_back(c);
            if (c == ' ' || c == '\n' || c == '\t') {
                // count word boundary only when next char is non-space
                if (i + 1 < s.size() && s[i+1] != ' ' && s[i+1] != '\n' && s[i+1] != '\t') {
                    ++words;
                    if (words >= max_words) break;
                }
            }
        }
        // trim trailing spaces
        while (!out.empty() && isspace((unsigned char)out.back())) out.pop_back();
        if (out.size() > 0 && (int)out.size() > max_chars) out.resize(max_chars);
        return out;
    };

    while (fgets(line, sizeof(line), file)) {
        if (max_stories > 0 && stories_processed >= max_stories) break;
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 0) continue;

        if (strcmp(line, "<|endoftext|>") == 0) {
            if (current_story.empty()) continue;

            // tokenize story
            int num_tokens = 0;
            // allocate a safe per-story token buffer sized to worst-case:
            // worst-case: each byte becomes a token (+prefix tokens). reserve at least seq_len+4.
            int safe_capacity = std::max((int)current_story.length() * 2 + 16, seq_len + 4);
            std::vector<int> tokens_buf(safe_capacity);
            char *dup = strdup(current_story.c_str());
            encode(tokenizer, dup, 1, 0, tokens_buf.data(), &num_tokens);
            free(dup);
            // Defensive check: encode shouldn't write past safe_capacity. If it did, clamp.
            if (num_tokens > safe_capacity) {
                fprintf(stderr, "Warning: encode produced %d tokens but buffer capacity was %d; clamping\n",
                        num_tokens, safe_capacity);
                num_tokens = safe_capacity;
            }
            if (num_tokens < 2) { current_story.clear(); continue; }

            // Clear device KV cache for this story
            bos.key_bo.write(zero_cache.data(), cache_dim * sizeof(float), 0); bos.key_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.value_bo.write(zero_cache.data(), cache_dim * sizeof(float), 0); bos.value_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Per-story accumulators
            double story_logprob_sum = 0.0;
            int story_token_count = 0;
            long story_total_infer_ms = 0;
            long story_first_token_ms = 0;

            std::string preview = first_words(current_story, 6, 120);

            // per-token inference loop
            for (int local_pos = 0; local_pos < num_tokens - 1; ++local_pos) {
              // ensure we do not exceed kernel seq_len (kernel expects pos < seq_len)
                if (local_pos >= seq_len) {
                  fprintf(stderr, "Warning: story %d has > seq_len (%d) tokens; truncating at %d\n",
                          stories_processed + 1, seq_len, seq_len);
                  break;
                }
                int current_token = tokens_buf[local_pos];
                int target_token = tokens_buf[local_pos + 1];
                if (target_token < 0 || target_token >= vocab_size) {
                    fprintf(stderr, "Warning: invalid target token id %d (story %d pos %d). Skipping.\n",
                            target_token, stories_processed + 1, local_pos);
                    continue;
                }

                long inf_start = time_in_ms();

                auto run = kernel(
                  bos.emb_bo,
                  bos.wq_bo, bos.wq_s_bo,
                  bos.wk_bo, bos.wk_s_bo,
                  bos.wv_bo, bos.wv_s_bo,
                  bos.wo_bo, bos.wo_s_bo,
                  bos.w1_bo, bos.w1_s_bo,
                  bos.w2_bo, bos.w2_s_bo,
                  bos.w3_bo, bos.w3_s_bo,
                  bos.rms_att_bo,
                  bos.rms_ffn_bo,
                  bos.rms_final_bo,
                  bos.wcls_bo, bos.wcls_s_bo,
                  bos.key_bo,
                  bos.value_bo,
                  bos.out_bo,
                  current_token,
                  local_pos
                );
                run.wait();

                long inf_end = time_in_ms();
                long inf_time = inf_end - inf_start;
                if (local_pos == 0) story_first_token_ms = inf_time;
                story_total_infer_ms += inf_time;

                bos.out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                bos.out_bo.read(logits.data(), vocab_size * sizeof(float), 0);

                // accumulate log-prob for this token
                float lp = compute_log_prob(logits.data(), target_token, vocab_size);
                story_logprob_sum += lp;
                story_token_count++;

                // update running eval
            }

            // compute per-story metrics
            float story_perplexity = -1.0f;
            if (story_token_count > 0) {
                float avg_nll = - (float)(story_logprob_sum / story_token_count);
                story_perplexity = expf(avg_nll);
            }
            float story_throughput = (story_total_infer_ms > 0 && story_token_count > 0) ?
                                     (story_token_count / (float)story_total_infer_ms * 1000.0f) : 0.0f;

            // update running eval
            eval.total_log_prob += (float)story_logprob_sum;
            eval.total_tokens += story_token_count;
            eval.total_inference_time_ms += story_total_infer_ms;
            eval.total_first_token_time_ms += story_first_token_ms;
            eval.total_stories++;
            eval.total_inference_calls += story_token_count;
            if (story_throughput > 0.0f) eval.story_throughputs.push_back(story_throughput);
            eval.story_first_token_latencies.push_back(story_first_token_ms);
            eval.story_token_counts.push_back(story_token_count);

            // running metrics
            float running_perplexity = calculate_final_perplexity(&eval);
            float running_throughput = (eval.total_inference_time_ms > 0) ?
                                       (eval.total_tokens / (float)eval.total_inference_time_ms * 1000.0f) : 0.0f;

            // Print concise per-story report
            printf("Story %d: preview=\"%s\"\n", stories_processed + 1, preview.c_str());
            printf("  tokens=%d  perplexity=%.4f  throughput=%.2f tok/s  first_token=%ld ms\n",
                   story_token_count, story_perplexity, story_throughput, story_first_token_ms);
            printf("  running: stories=%d  tokens=%d  perplexity=%.4f  throughput=%.2f tok/s\n\n",
                   eval.total_stories, eval.total_tokens, running_perplexity, running_throughput);

            stories_processed++;
            current_story.clear();
        } else {
            if (!current_story.empty()) current_story += " ";
            current_story += line;
        }
    }

    long overall_end = time_in_ms();
    fclose(file);

    fprintf(stderr, "Processed %d stories in %ld ms\n", stories_processed, overall_end - overall_start);

    // print final summary
    float final_perplexity = calculate_final_perplexity(&eval);
    float overall_throughput = (eval.total_inference_time_ms > 0) ?
                               (eval.total_tokens / (float)eval.total_inference_time_ms * 1000.0f) : 0.0f;
    fprintf(stderr, "FINAL: stories=%d tokens=%d perplexity=%.4f throughput=%.2f tok/s avg_first_token_ms=%.2f\n",
            eval.total_stories, eval.total_tokens, final_perplexity, overall_throughput,
            (eval.total_stories > 0) ? (eval.total_first_token_time_ms / (double)eval.total_stories) : 0.0);
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
  Weights weights;
  read_checkpoint(checkpoint_path, &weights);
  if (steps <= 0 || steps > seq_len)
    steps = seq_len;

  // Build Tokenizer
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

  // Build Sampler
  Sampler sampler;
  build_sampler(&sampler, vocab_size, temperature, topp, rng_seed);

  // Run generation or other modes
  if (mode == "generate") {
    generate(&weights, &tokenizer, &sampler, prompt, steps, kernelpath);
  } else if (mode == "evaluate") {
    evaluate(eval_file, &weights, &tokenizer, kernelpath);  
  } else {
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
