#pragma once

#ifndef FORWARD_H
#define FORWARD_H

#include "typedefs.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <hls_math.h>


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

void rope_rotation(float q[dim], float k[kv_dim], float q_out[dim], float k_out[kv_dim], int pos);

void multihead_attention_with_cache(float q[dim], float k[kv_dim], float v[kv_dim],
                                    float *key_cache, float *value_cache,
                                    float xb[dim], float att[n_heads * seq_len],
                                    int layer, int pos);

void swiglu_activation(float hb[hidden_dim], float hb2[hidden_dim], float o[hidden_dim]);

template<int SIZE>
void QKV_matmul_rotation(int pos, float q[dim], float k[kv_dim], float v[kv_dim], 
                         int8_t xq_q[dim], float xq_s[dim/GS],
                         int8_t *wq_weights, float *wq_scales,
                         int8_t *wk_weights, float *wk_scales,
                         int8_t *wv_weights, float *wv_scales);

void FFN_matmul(float o[hidden_dim], 
                int8_t xq_q[dim], float xq_s[dim/GS],
                int8_t *w1_weights, float *w1_scales,
                int8_t *w3_weights, float *w3_scales);



// ============================================================================
// TEMPLATE HELPER FUNCTIONS
// ============================================================================

template<int SIZE>
void residual_add(float x[SIZE], float residual[SIZE]) {
    
    residual_loop:
    for (int i = 0; i < SIZE; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        x[i] += residual[i];
    }
}

// TODO: Look into array partitioning for loop unrolling
template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
    
    float w_buffer[S];
    #pragma HLS ARRAY_PARTITION variable=w_buffer type=cyclic factor=64
    rms_buffer:
    for (int i = 0; i < S; i ++) {
        #pragma HLS PIPELINE II=1
        w_buffer[i] = weight[i];
    }

    // Calculate sum of squares
    float ss = 0.0f;
    sum_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=64
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        ss += x[j] * x[j];
    }

    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / hls::sqrtf(ss);
    
    // Normalize and scale
    normalize:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=64
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        o[j] = w_buffer[j] * (ss * x[j]);
    }
}

// TODO: Look into find_max/exp_sum reduction and loop-carried dependency optimizations
template<int SIZE>
void softmax(float *x, int size) {
    // Find max
    float max_val = x[0];
    find_max:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE 
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

        if (x[i] > max_val) max_val = x[i];
    }
    
    // Exp and sum
    float sum = 0.0f;
    exp_sum:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

        x[i] = hls::expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    normalize:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

        x[i] /= sum;
    }
}

// TODO: Figure out optimization, multirow or multicol
// Matrix * vector, wq (D*N) by xq (N elements) -> xout (D)
template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
    
    matmul_outer:
    for (int i = 0; i < D; i++) {

        float val = 0.0f;

        int8_t wq_buffer[N];
        float ws_buffer[N/GS];

        #pragma HLS ARRAY_PARTITION variable=wq_buffer type=cyclic factor=32
        #pragma HLS ARRAY_PARTITION variable=ws_buffer type=cyclic factor=32

        // Initialize wq buffer (N) for each row of wq (D*N)
        matmul_wq_buffer:
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            wq_buffer[j] = wq[j + (i * N)];
        }

        // Initialize ws buffer (N/GS) for each row of wq (D*N/GS)
        matmul_ws_buffer:
        for (int j = 0; j < N/GS; j++) {
            #pragma HLS PIPELINE II=1
            ws_buffer[j] = ws[(i * N / GS) + j];
        }

        matmul_groups:
        for (int j = 0; j <= N - GS; j += GS) {
            #pragma HLS PIPELINE

            int32_t dot = 0;

            dot_product:
            for (int k = 0; k < GS; k++) {
                #pragma HLS UNROLL factor=32

                    dot += ((int32_t)xq[j+k]) * ((int32_t)wq_buffer[j+k]);
            }
            val += ((float)dot) * ws_buffer[j / GS] * xs[j / GS];
        }
        xout[i] = val;
    }
}


// TODO: find max reduction work around, causing slack
template<int S>
void quantize(int8_t qx_q[S], float qx_s[S/GS], float x[S]) {
    
    
    constexpr int num_groups = S / GS;
    constexpr float inv_Q_MAX = 1.0f / 127.0f;
    
    main_loop:
    for (int i = 0; i < S/GS; i++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=64
        #pragma HLS LOOP_TRIPCOUNT min=12 max=32

        
        // Find max absolute value in group
        float wmax = 0.0f;
        find_max:
        for (int j = 0; j < GS; j++) {
            float val = std::abs(x[i * GS + j]);
            if (val > wmax) wmax = val;
        }
        
        // Calculate scale and quantize
        float scale = wmax * inv_Q_MAX;
        qx_s[i] = scale;
        
        float inv_scale = 1.0f / scale;
        
        quantize_group:
        for (int j = 0; j < GS; j++) {
            float quant_val = x[i * GS + j] * inv_scale;
            qx_q[i * GS + j] = (int8_t)quant_val;
        }
    }
}

// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================

void attention_block(
    int layer, int pos,
    float x[dim], float x_out[dim],
    float att[n_heads * seq_len],
    int8_t *wq_weights, float *wq_scales,
    int8_t *wk_weights, float *wk_scales,
    int8_t *wv_weights, float *wv_scales,
    int8_t *wo_weights, float *wo_scales,
    float *rms_att_weight,
    float *key_cache, float *value_cache
);

void ffn_block(
    int layer,
    float x[dim], float x_out[dim],
    int8_t *w1_weights, float *w1_scales,
    int8_t *w2_weights, float *w2_scales,
    int8_t *w3_weights, float *w3_scales,
    float *rms_ffn_weight

);

void final_classifier(
    float x[dim], float *out,
    float *rms_final_weight,
    int8_t *wcls_weights, float *wcls_scales
);



#endif // FORWARD_H