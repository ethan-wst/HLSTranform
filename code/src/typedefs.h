#pragma once
#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <stdint.h>
#include <stdio.h>

//===========================================================================
//  typedefs.h
//===========================================================================
//  @brief: Core type definitions for Llama 2 transformer

struct Config
{
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len;    // max sequence length
  int GS;         // group size for quantization
};

#endif // TYPEDEFS_H