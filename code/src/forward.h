#pragma once

#include "typedefs.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>

//===========================================================================
// forward.h
//===========================================================================
// @brief: Forward pass function declarations for Llama 2 transformer
//         with FLATTENED INTERFACE for optimal HBM access

// ============================================================================
// TOP-LEVEL FORWARD FUNCTION
// ============================================================================
// All weight arrays are passed individually, allowing each to be mapped to 
// separate HBM banks. This design enables:
//   - Parallel weight access across up to 23 HBM banks
//   - Optimal burst patterns for sequential array access
//   - Independent bandwidth optimization per weight type
//   - Proper depth specification for each m_axi interface

extern "C" void forward(
    // Embedding Weights
    float *token_embedding_table,     // [vocab_size * dim] - bundle=gmem0
    
    // Query weights
    int8_t *wq_weights,               // [n_layers * dim * dim] - bundle=gmem1
    float *wq_scales,                 // [n_layers * dim * dim / GS] - bundle=gmem2
    
    // Key weights
    int8_t *wk_weights,               // [n_layers * dim * kv_dim] - bundle=gmem3
    float *wk_scales,                 // [n_layers * dim * kv_dim / GS] - bundle=gmem4
    
    // Value weights
    int8_t *wv_weights,               // [n_layers * dim * kv_dim] - bundle=gmem5
    float *wv_scales,                 // [n_layers * dim * kv_dim / GS] - bundle=gmem6
    
    // Output projection weights
    int8_t *wo_weights,               // [n_layers * dim * dim] - bundle=gmem7
    float *wo_scales,                 // [n_layers * dim * dim / GS] - bundle=gmem8
    
    // FFN layer 1 (expansion)
    int8_t *w1_weights,               // [n_layers * dim * hidden_dim] - bundle=gmem9
    float *w1_scales,                 // [n_layers * dim * hidden_dim / GS] - bundle=gmem10
    
    // FFN layer 2 (contraction)
    int8_t *w2_weights,               // [n_layers * hidden_dim * dim] - bundle=gmem11
    float *w2_scales,                 // [n_layers * hidden_dim * dim / GS] - bundle=gmem12
    
    // FFN layer 3 (gating)
    int8_t *w3_weights,               // [n_layers * dim * hidden_dim] - bundle=gmem13
    float *w3_scales,                 // [n_layers * dim * hidden_dim / GS] - bundle=gmem14
    
    float *rms_att_weight,            // [n_layers * dim] - bundle=gmem15
    float *rms_ffn_weight,            // [n_layers * dim] - bundle=gmem16
    float *rms_final_weight,          // [dim] - bundle=gmem17
    
    int8_t *wcls_weights,             // [vocab_size * dim] - bundle=gmem18
    float *wcls_scales,               // [vocab_size * dim / GS] - bundle=gmem19
    
    float *key_cache,                 // [n_layers * seq_len * kv_dim] - bundle=gmem20
    float *value_cache,               // [n_layers * seq_len * kv_dim] - bundle=gmem21
    
    float *out,                       // [vocab_size] - bundle=gmem22
    
    int token,                        // Current input token
    int pos                          // Position in sequence
);


// HELPER FUNCTION DECLARATIONS

// These functions are called from the top-level forward() function
// Note: m_axi pragmas are REMOVED from helpers in dataflow design
//       Only the top-level function should have m_axi interfaces

// RMS Normalization
template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]);

// Matrix multiplication with quantized weights
template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);

// Softmax activation
template<int SIZE>
void softmax(float *x, int size);


// TEMPLATE IMPLEMENTATIONS (inline in header)

// Dequantize a quantized tensor into float array
template<int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    dequant_loop:
    for (int i = 0; i < S; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

// Quantize float array into quantized tensor
template<int S>
void quantize(QuantizedTensor<S> *qx, float x[S]) {
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
        qx->s[group] = scale;
        
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        
        quantize_group:
        for (int i = 0; i < GS; i++) {
            float quant_val = x[group * GS + i] * inv_scale;
            qx->q[group * GS + i] = (int8_t)quant_val;
        }
    }
}

#endif