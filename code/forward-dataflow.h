#pragma once

#include "typedefs.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>

//===========================================================================
// forward.h - DATAFLOW READY VERSION
//===========================================================================
// Function declarations for Llama 2 transformer with extracted helper functions

// ============================================================================
// TOP-LEVEL FORWARD FUNCTION
// ============================================================================

extern "C" void forward(
    // Embedding Weights
    float *token_embedding_table,
    
    // Attention Weights (all 12 layers)
    int8_t *wq_weights,
    float *wq_scales,
    int8_t *wk_weights,
    float *wk_scales,
    int8_t *wv_weights,
    float *wv_scales,
    int8_t *wo_weights,
    float *wo_scales,
    
    // FFN Weights (all 12 layers)
    int8_t *w1_weights,
    float *w1_scales,
    int8_t *w2_weights,
    float *w2_scales,
    int8_t *w3_weights,
    float *w3_scales,
    
    // RMS Normalization Weights
    float *rms_att_weight,
    float *rms_ffn_weight,
    float *rms_final_weight,
    
    // Classifier Weights
    int8_t *wcls_weights,
    float *wcls_scales,
    
    // KV Cache (all 12 layers)
    float *key_cache,
    float *value_cache,
    
    // Output
    float *out,
    
    // Control Parameters
    int token,
    int pos
);

// ============================================================================
// EXTRACTED HELPER FUNCTIONS FOR DATAFLOW
// ============================================================================

// Load token embedding from table
void load_embedding(float *token_embedding_table, float x[dim], int token);

// Apply RoPE rotation to Q and K
void rope_rotation(float q[dim], float k[kv_dim], int pos);

// Update KV cache for current layer and position
void update_kv_cache(float k[kv_dim], float v[kv_dim],
                     float *key_cache, float *value_cache,
                     int layer, int pos);

// Multi-head attention computation
void multihead_attention(float q[dim], 
                        float *key_cache, float *value_cache,
                        float xb[dim], float att[n_heads * seq_len],
                        int layer, int pos);

// Residual addition (template for flexibility)
template<int SIZE>
void residual_add(float x[SIZE], float residual[SIZE]);

// SwiGLU activation function
void swiglu_activation(float hb[hidden_dim], float hb2[hidden_dim]);

// High-level attention block wrapper
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

// High-level FFN block wrapper
void ffn_block(
    int layer,
    float x[dim], float xb[dim],
    float hb[hidden_dim], float hb2[hidden_dim],
    int8_t *w1_weights, float *w1_scales,
    int8_t *w2_weights, float *w2_scales,
    int8_t *w3_weights, float *w3_scales,
    float *rms_ffn_weight,
    QuantizedTensor<dim> *xq,
    QuantizedTensor<hidden_dim> *hq
);

// Final classifier after all layers
void final_classifier(
    float x[dim], float *out,
    float *rms_final_weight,
    int8_t *wcls_weights, float *wcls_scales,
    QuantizedTensor<dim> *xq
);

// ============================================================================
// CORE HELPER FUNCTIONS (Templates)
// ============================================================================

// RMS Normalization
template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]);

// Quantized matrix multiplication
template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);

// Softmax activation
template<int SIZE>
void softmax(float *x, int size);

// ============================================================================
// TEMPLATE IMPLEMENTATIONS (inline in header)
// ============================================================================

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

#endif // FORWARD_H
