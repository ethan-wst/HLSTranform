#include "forward.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>

// ============================================================================
// EXTRACTED HELPER FUNCTIONS FOR DATAFLOW
// ============================================================================

// Function 1: Load token embedding
void load_embedding(float *token_embedding_table, float x[dim], int token) {
    #pragma HLS INLINE off
    
    load_loop:
    for (int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        x[i] = token_embedding_table[token * dim + i];
    }
}

// Function 2: RoPE rotation for positional encoding
void rope_rotation(float q[dim], float k[kv_dim], int pos) {
    #pragma HLS INLINE off
    
    constexpr int kv_dim_local = (dim * n_kv_heads) / n_heads;
    constexpr int head_size = dim / n_heads;
    constexpr float inv_10000 = 1.0f / 10000.0f;
    const float inv_head_size = 1.0f / float(head_size);
    
    // Rotate both Q and K for kv_dim dimensions
    rotation1:
    for (int i = 0; i < kv_dim_local; i += 2) {
        #pragma HLS PIPELINE II=1
        
        int head_dim = i % head_size;
        float freq = hls::powf(inv_10000, head_dim * inv_head_size);
        float val = pos * freq;
        float fcr = hls::cosf(val);
        float fci = hls::sinf(val);
        
        // Rotate query
        float v0_q = q[i];
        float v1_q = q[i + 1];
        q[i] = v0_q * fcr - v1_q * fci;
        q[i + 1] = v0_q * fci + v1_q * fcr;
        
        // Rotate key
        float v0_k = k[i];
        float v1_k = k[i + 1];
        k[i] = v0_k * fcr - v1_k * fci;
        k[i + 1] = v0_k * fci + v1_k * fcr;
    }
    
    // Rotate only Q for remaining dimensions
    rotation2:
    for (int i = kv_dim_local; i < dim; i += 2) {
        #pragma HLS PIPELINE II=1
        
        int head_dim = i % head_size;
        float freq = hls::powf(inv_10000, head_dim * inv_head_size);
        float val = pos * freq;
        float fcr = hls::cosf(val);
        float fci = hls::sinf(val);
        
        float v0 = q[i];
        float v1 = q[i + 1];
        q[i] = v0 * fcr - v1 * fci;
        q[i + 1] = v0 * fci + v1 * fcr;
    }
}

// Function 3: Update KV cache
void update_kv_cache(float k[kv_dim], float v[kv_dim],
                     float *key_cache, float *value_cache,
                     int layer, int pos) {
    #pragma HLS INLINE off
    
    constexpr int kv_dim_local = (dim * n_kv_heads) / n_heads;
    const int kv_cache_layer_offset = layer * seq_len * kv_dim_local;
    const int kv_cache_pos_offset = kv_cache_layer_offset + pos * kv_dim_local;
    
    update_key:
    for (int i = 0; i < kv_dim_local; i++) {
        #pragma HLS PIPELINE II=1
        key_cache[kv_cache_pos_offset + i] = k[i];
    }
    
    update_value:
    for (int i = 0; i < kv_dim_local; i++) {
        #pragma HLS PIPELINE II=1
        value_cache[kv_cache_pos_offset + i] = v[i];
    }
}

// Function 4: Multi-head attention
void multihead_attention(float q[dim], 
                        float *key_cache, float *value_cache,
                        float xb[dim], float att[n_heads * seq_len],
                        int layer, int pos) {
    #pragma HLS INLINE off
    
    constexpr int kv_dim_local = (dim * n_kv_heads) / n_heads;
    constexpr int head_size = dim / n_heads;
    const int kv_cache_layer_offset = layer * seq_len * kv_dim_local;
    
    multihead_loop:
    for (int h = 0; h < n_heads; h++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off
        
        float *q_head = q + h * head_size;
        float *att_head = att + h * seq_len;
        
        // Compute attention scores for this head
        att_scores:
        for (int t = 0; t <= pos; t++) {
            #pragma HLS PIPELINE II=1
            int kv_head = h / (n_heads / n_kv_heads);
            float *k_head = &key_cache[kv_cache_layer_offset + t * kv_dim_local + kv_head * head_size];
            
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q_head[i] * k_head[i];
            }
            att_head[t] = score / hls::sqrtf((float)head_size);
        }
        
        // Softmax over attention scores
        softmax<seq_len>(att_head, pos + 1);
        
        // Weighted sum of values
        float *xb_head = xb + h * head_size;
        
        init_xb:
        for (int i = 0; i < head_size; i++) {
            #pragma HLS PIPELINE II=1
            xb_head[i] = 0.0f;
        }
        
        att_weighted_sum:
        for (int t = 0; t <= pos; t++) {
            #pragma HLS PIPELINE off
            int kv_head = h / (n_heads / n_kv_heads);
            float *v_head = &value_cache[kv_cache_layer_offset + t * kv_dim_local + kv_head * head_size];
            float a = att_head[t];
            
            for (int i = 0; i < head_size; i++) {
                #pragma HLS PIPELINE II=1
                xb_head[i] += a * v_head[i];
            }
        }
    }
}

// Function 5: Residual addition (template for flexibility)
template<int SIZE>
void residual_add(float x[SIZE], float residual[SIZE]) {
    #pragma HLS INLINE off
    
    add_loop:
    for (int i = 0; i < SIZE; i++) {
        #pragma HLS PIPELINE II=1
        x[i] += residual[i];
    }
}

// Function 6: SwiGLU activation
void swiglu_activation(float hb[hidden_dim], float hb2[hidden_dim]) {
    #pragma HLS INLINE off
    
    swiglu_loop:
    for (int i = 0; i < hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
        float val = hb[i];
        val *= (1.0f / (1.0f + hls::expf(-val)));  // SiLU/Swish
        val *= hb2[i];                              // Gated
        hb[i] = val;
    }
}

// Function 7: Attention block (high-level wrapper)
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
) {
    #pragma HLS INLINE off
    // Future: Can add #pragma HLS DATAFLOW here for sub-function pipelining
    
    const int dim_dim_offset = layer * dim * dim;
    const int dim_kv_offset = layer * dim * kv_dim;
    const int rms_offset = layer * dim;
    
    // Step 1: Attention preprocessing
    rmsnorm<dim>(xb, x, &rms_att_weight[rms_offset]);
    quantize<dim>(xq, xb);
    
    // Step 2: QKV projections
    matmul<dim, dim>(q, xq->q, xq->s, &wq_weights[dim_dim_offset], &wq_scales[dim_dim_offset / GS]);
    matmul<kv_dim, dim>(k, xq->q, xq->s, &wk_weights[dim_kv_offset], &wk_scales[dim_kv_offset / GS]);
    matmul<kv_dim, dim>(v, xq->q, xq->s, &wv_weights[dim_kv_offset], &wv_scales[dim_kv_offset / GS]);
    
    // Step 3: RoPE and cache
    rope_rotation(q, k, pos);
    update_kv_cache(k, v, key_cache, value_cache, layer, pos);
    
    // Step 4: Multi-head attention
    multihead_attention(q, key_cache, value_cache, xb, att, layer, pos);
    
    // Step 5: Output projection
    quantize<dim>(xq, xb);
    matmul<dim, dim>(xb2, xq->q, xq->s, &wo_weights[dim_dim_offset], &wo_scales[dim_dim_offset / GS]);
    
    // Step 6: Residual
    residual_add<dim>(x, xb2);
}

// Function 8: FFN block (high-level wrapper)
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
) {
    #pragma HLS INLINE off
    // Future: Can add #pragma HLS DATAFLOW here for sub-function pipelining
    
    const int dim_hidden_offset = layer * dim * hidden_dim;
    const int hidden_dim_offset = layer * hidden_dim * dim;
    const int rms_offset = layer * dim;
    
    // Step 1: FFN preprocessing
    rmsnorm<dim>(xb, x, &rms_ffn_weight[rms_offset]);
    quantize<dim>(xq, xb);
    
    // Step 2: FFN forward (SwiGLU)
    matmul<hidden_dim, dim>(hb, xq->q, xq->s, &w1_weights[dim_hidden_offset], &w1_scales[dim_hidden_offset / GS]);
    matmul<hidden_dim, dim>(hb2, xq->q, xq->s, &w3_weights[dim_hidden_offset], &w3_scales[dim_hidden_offset / GS]);
    swiglu_activation(hb, hb2);
    
    // Step 3: Project back to dim
    quantize<hidden_dim>(hq, hb);
    matmul<dim, hidden_dim>(xb, hq->q, hq->s, &w2_weights[hidden_dim_offset], &w2_scales[hidden_dim_offset / GS]);
    
    // Step 4: Residual
    residual_add<dim>(x, xb);
}

// Function 9: Final classifier
void final_classifier(
    float x[dim], float *out,
    float *rms_final_weight,
    int8_t *wcls_weights, float *wcls_scales,
    QuantizedTensor<dim> *xq
) {
    #pragma HLS INLINE off
    
    rmsnorm<dim>(x, x, rms_final_weight);
    quantize<dim>(xq, x);
    matmul<vocab_size, dim>(out, xq->q, xq->s, wcls_weights, wcls_scales);
}

// ============================================================================
// TOP-LEVEL FORWARD FUNCTION - DATAFLOW READY
// ============================================================================

extern "C" void forward(
    // Embedding weights
    float *token_embedding_table,
    
    // Attention weights
    int8_t *wq_weights,
    float *wq_scales,
    int8_t *wk_weights,
    float *wk_scales,
    int8_t *wv_weights,
    float *wv_scales,
    int8_t *wo_weights,
    float *wo_scales,
    
    // FFN weights
    int8_t *w1_weights,
    float *w1_scales,
    int8_t *w2_weights,
    float *w2_scales,
    int8_t *w3_weights,
    float *w3_scales,
    
    // RMS norm weights
    float *rms_att_weight,
    float *rms_ffn_weight,
    float *rms_final_weight,
    
    // Classifier weights
    int8_t *wcls_weights,
    float *wcls_scales,
    
    // KV cache
    float *key_cache,
    float *value_cache,
    
    // Output
    float *out,
    
    // Control parameters
    int token,
    int pos
) {
    // ========================================================================
    // INTERFACE PRAGMAS
    // ========================================================================
    
    #pragma HLS INTERFACE m_axi port=token_embedding_table offset=slave depth=24576000 bundle=gmem0 max_read_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=wq_weights offset=slave depth=7077888 bundle=gmem1 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wq_scales offset=slave depth=110592 bundle=gmem2 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wk_weights offset=slave depth=7077888 bundle=gmem3 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wk_scales offset=slave depth=110592 bundle=gmem4 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wv_weights offset=slave depth=7077888 bundle=gmem5 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wv_scales offset=slave depth=110592 bundle=gmem6 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wo_weights offset=slave depth=7077888 bundle=gmem7 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wo_scales offset=slave depth=110592 bundle=gmem8 max_read_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=w1_weights offset=slave depth=18874368 bundle=gmem9 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w1_scales offset=slave depth=294912 bundle=gmem10 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w2_weights offset=slave depth=18874368 bundle=gmem11 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w2_scales offset=slave depth=294912 bundle=gmem12 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w3_weights offset=slave depth=18874368 bundle=gmem13 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=w3_scales offset=slave depth=294912 bundle=gmem14 max_read_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=rms_att_weight offset=slave depth=9216 bundle=gmem15 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=rms_ffn_weight offset=slave depth=9216 bundle=gmem16 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=rms_final_weight offset=slave depth=768 bundle=gmem17 max_read_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=wcls_weights offset=slave depth=24576000 bundle=gmem18 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=wcls_scales offset=slave depth=384000 bundle=gmem19 max_read_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=key_cache offset=slave depth=9437184 bundle=gmem20 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=value_cache offset=slave depth=9437184 bundle=gmem21 max_read_burst_length=256 max_write_burst_length=256
    
    #pragma HLS INTERFACE m_axi port=out offset=slave depth=32000 bundle=gmem22 max_write_burst_length=256
    
    #pragma HLS INTERFACE s_axilite port=token bundle=control
    #pragma HLS INTERFACE s_axilite port=pos bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // ========================================================================
    // LOCAL ARRAYS - Communication between functions
    // ========================================================================
    
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    
    // Main activation buffers
    float x[dim];
    float xb[dim];
    float xb2[dim];
    float hb[hidden_dim];
    float hb2[hidden_dim];
    float q[dim];
    float k[kv_dim];
    float v[kv_dim];
    float att[n_heads * seq_len];
    
    // Quantized tensors
    QuantizedTensor<dim> xq;
    QuantizedTensor<hidden_dim> hq;
    
    // ========================================================================
    // FORWARD PASS - ONLY FUNCTION CALLS (Dataflow Canonical Form)
    // ========================================================================
    
    // Load embedding (once at start)
    load_embedding(token_embedding_table, x, token);
    
    // Main layer loop - clean function calls only
    main_forward_loop:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12
        
        // Attention block
        attention_block(l, pos, x, xb, xb2, q, k, v, att,
                       wq_weights, wq_scales, wk_weights, wk_scales,
                       wv_weights, wv_scales, wo_weights, wo_scales,
                       rms_att_weight, key_cache, value_cache, &xq);
        
        // FFN block
        ffn_block(l, x, xb, hb, hb2,
                 w1_weights, w1_scales, w2_weights, w2_scales,
                 w3_weights, w3_scales, rms_ffn_weight, &xq, &hq);
    }
    
    // Final classifier
    final_classifier(x, out, rms_final_weight, wcls_weights, wcls_scales, &xq);
}

// ============================================================================
// EXISTING HELPER FUNCTION IMPLEMENTATIONS (rmsnorm, softmax, matmul)
// ============================================================================

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
