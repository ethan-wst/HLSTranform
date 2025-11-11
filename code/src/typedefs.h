#include <stdint.h>
#include <stdio.h>

// TODO: replace with HLS types (vector, int, fp, etc...)

#ifndef TYPEDEFS
#define TYPEDEFS

constexpr float inv_10000 = 1.0f / 10000.0f;

// Configuration structure for transformer hyperparameters
struct Config {
    int dim;          // transformer dimension
    int hidden_dim;   // for ffn layers
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;   // vocabulary size, usually 256 (byte-level)
    int seq_len;      // max sequence length
    int GS;           // group size for quantization
};

#endif