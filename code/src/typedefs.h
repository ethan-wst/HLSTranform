#include <stdint.h>
#include <stdio.h>

// TODO: replace with HLS types (vector, int, fp, etc...)

//===========================================================================
//  typedefs.h
//===========================================================================
//  @brief: This header defines the shorthand of several ap_uint data types.

#ifndef TYPEDEFS
#define TYPEDEFS

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

// Quantized tensor structure: int8 quantized values with per-group float scales
template<int SIZE>
struct QuantizedTensor {
    int8_t q[SIZE];      // quantized values
    float s[SIZE / 64];  // scaling factors (one per group, assuming GS=64)
};

#endif
