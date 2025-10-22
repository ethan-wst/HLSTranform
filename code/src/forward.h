#pragma once

#ifndef FORWARD_H
#define FORWARD_H

#include "typedefs.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <hls_math.h>

// Compile Time Constant
constexpr int kv_dim = (dim * n_kv_heads) / n_heads;

// ============================================================================
// TOP-LEVEL FORWARD FUNCTION
// ============================================================================

extern "C" void forward(
    // Embedding Weights
    float *token_embedding_table,     // [vocab_size * dim] - bundle=gmem0
    
    // Attention Weights (all 12 layers)
    int8_t *wq_weights,               // [n_layers * dim * dim] - bundle=gmem1
    float *wq_scales,                 // [n_layers * dim * dim / GS] - bundle=gmem2
    int8_t *wk_weights,               // [n_layers * dim * kv_dim] - bundle=gmem3
    float *wk_scales,                 // [n_layers * dim * kv_dim / GS] - bundle=gmem4
    int8_t *wv_weights,               // [n_layers * dim * kv_dim] - bundle=gmem5
    float *wv_scales,                 // [n_layers * dim * kv_dim / GS] - bundle=gmem6
    int8_t *wo_weights,               // [n_layers * dim * dim] - bundle=gmem7
    float *wo_scales,                 // [n_layers * dim * dim / GS] - bundle=gmem8
    
    // FFN Weights (all 12 layers)
    int8_t *w1_weights,               // [n_layers * dim * hidden_dim] - bundle=gmem9
    float *w1_scales,                 // [n_layers * dim * hidden_dim / GS] - bundle=gmem10
    int8_t *w2_weights,               // [n_layers * hidden_dim * dim] - bundle=gmem11
    float *w2_scales,                 // [n_layers * hidden_dim * dim / GS] - bundle=gmem12
    int8_t *w3_weights,               // [n_layers * dim * hidden_dim] - bundle=gmem13
    float *w3_scales,                 // [n_layers * dim * hidden_dim / GS] - bundle=gmem14
    
    // RMS Normalization Weights
    float *rms_att_weight,            // [n_layers * dim] - bundle=gmem15
    float *rms_ffn_weight,            // [n_layers * dim] - bundle=gmem16
    float *rms_final_weight,          // [dim] - bundle=gmem17
    
    // Classifier Weights
    int8_t *wcls_weights,             // [vocab_size * dim] - bundle=gmem18
    float *wcls_scales,               // [vocab_size * dim / GS] - bundle=gmem19
    
    // KV Cache (all 12 layers)
    float *key_cache,                 // [n_layers * seq_len * kv_dim] - bundle=gmem20
    float *value_cache,               // [n_layers * seq_len * kv_dim] - bundle=gmem21
    
    // Output
    float *out,                       // [vocab_size] - bundle=gmem22
    
    // Control Parameters
    int token,
    int pos
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void load_embedding(float *token_embedding_table, float x[dim], int token);

void rope_rotation(float q[dim], float k[kv_dim], int pos);

void update_kv_cache(float k[kv_dim], float v[kv_dim],
                     float *key_cache, float *value_cache,
                     int layer, int pos);

void multihead_attention(float q[dim], 
                        float *key_cache, float *value_cache,
                        float xb[dim], float att[n_heads * seq_len],
                        int layer, int pos);

void swiglu_activation(float hb[hidden_dim], float hb2[hidden_dim]);

// ============================================================================
// TEMPLATE HELPER FUNCTIONS
// ============================================================================

template<int SIZE>
void residual_add(float x[SIZE], float residual[SIZE]) {
    #pragma HLS INLINE off
    
    add_loop:
    for (int i = 0; i < SIZE; i++) {
        #pragma HLS PIPELINE II=1
        x[i] += residual[i];
    }
}

template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
    #pragma HLS INLINE off
    
    // Calculate sum of squares
    float ss = 0.0f;
    sum_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        ss += x[j] * x[j];
    }
    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / hls::sqrtf(ss);
    
    // Normalize and scale
    normalize:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        o[j] = weight[j] * (ss * x[j]);
    }
}

// TODO: Look into find_max/exp_sum reduction and loop-carried dependency optimizations
template<int SIZE>
void softmax(float *x, int size) {
    #pragma HLS INLINE off
    
    // Find max
    float max_val = x[0];
    find_max:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Exp and sum
    float sum = 0.0f;
    exp_sum:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        x[i] = hls::expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    normalize:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        x[i] /= sum;
    }
}

// TODO: Optimize further, ensure optimizations are sound
template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
    #pragma HLS INLINE off
    
    outer:
    for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE
        
        float val = 0.0f;
        
        inner:
        for (int j = 0; j <= N - GS; j += GS) {
            #pragma HLS UNROLL factor=4 skip_exit_check
            
            int32_t ival = 0;
            
            dot:
            for (int k = 0; k < GS; k++) {
                #pragma HLS UNROLL
                ival += ((int32_t)xq[j + k]) * ((int32_t)wq[i * N + j + k]);
            }
            
            val += ((float)ival) * ws[i * N / GS + j / GS] * xs[j / GS];
        }
        
        xout[i] = val;
    }
}

template<int S>
void quantize(int8_t qx_q[S], float qx_s[S/GS], float x[S]) {
    #pragma HLS INLINE off
    
    constexpr int num_groups = S / GS;
    constexpr float inv_Q_MAX = 1.0f / 127.0f;
    
    main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS PIPELINE
        
        // Find max absolute value in group
        float wmax = 0.0f;
        find_max:
        for (int i = 0; i < GS; i++) {
            float val = std::abs(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        
        // Calculate scale and quantize
        float scale = wmax * inv_Q_MAX;
        qx_s[group] = scale;
        
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        
        quantize_group:
        for (int i = 0; i < GS; i++) {
            float quant_val = x[group * GS + i] * inv_scale;
            qx_q[group * GS + i] = (int8_t)quant_val;
        }
    }
}

// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================

void attention_block(
    int layer, int pos,
    float x[dim], float xb[dim], float xb2[dim],
    float q[dim], float k[kv_dim], float v[kv_dim],
    float att[n_heads * seq_len],
    int8_t *wq_weights, float *wq_scales,
    int8_t *wk_weights, float *wk_scales,
    int8_t *wv_weights, float *wv_scales,
    int8_t *wo_weights, float *wo_scales,
    float *rms_att_weight,
    float *key_cache, float *value_cache,
    QuantizedTensor<dim> *xq
);

void ffn_block(
    int layer,
    float x[dim], float xb[dim],
    float hb[hidden_dim], float hb2[hidden_dim],
    int8_t *w1_weights, float *w1_scales,
    int8_t *w2_weights, float *w2_scales,
    int8_t *w3_weights, float *w3_scales,
    float *rms_ffn_weight,
    QuantizedTensor<dim> *xq,
    QuantizedTensor<dim> *hq
);

void final_classifier(
    float x[dim], float *out,
    float *rms_final_weight,
    int8_t *wcls_weights, float *wcls_scales,
    QuantizedTensor<dim> *xq
);



#endif // FORWARD_H