#include "forward.h"
#include "config.h"
#include "hls_stream.h"
#include <cstring>
#include <cmath>
#include <hls_math.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Load token embedding from lookup table
void load_embedding(float *token_embedding_table, float x[dim], int token) {

    load_loop:
    for (int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        x[i] = token_embedding_table[token * dim + i];
    }
}

// ATTENTION BLOCK HELPERS =====================================================

void rope_rotation(float q_in[dim], float k_in[kv_dim], float q_out[dim], float k_out[kv_dim], int pos) {

    const float inv_head_size = 1.0f / float(head_size);
    
    // Rotate both Q and K for kv_dim dimensions
    rotation1:
    for (int i = 0; i < kv_dim; i += 2) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor = 8
        #pragma HLS LOOP_TRIPCOUNT min=384 max=384
        
        int head_dim = i % head_size;
        float freq = hls::powf(inv_10000, head_dim * inv_head_size);
        float val = pos * freq;
        float fcr = hls::cosf(val);
        float fci = hls::sinf(val);
        
        // Rotate query
        float v0_q = q_in[i];
        float v1_q = q_in[i + 1];
        q_out[i] = v0_q * fcr - v1_q * fci;
        q_out[i + 1] = v0_q * fci + v1_q * fcr;
        
        // Rotate key
        float v0_k = k_in[i];
        float v1_k = k_in[i + 1];
        k_out[i] = v0_k * fcr - v1_k * fci;
        k_out[i + 1] = v0_k * fci + v1_k * fcr;
    }
    
    // Rotate only Q for remaining dimensions (MQA case)
    rotation2:
    for (int i = kv_dim; i < dim; i += 2) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor = 8
        #pragma HLS LOOP_TRIPCOUNT min=0 max=384
        
        int head_dim = i % head_size;
        float freq = hls::powf(inv_10000, head_dim * inv_head_size);
        float val = pos * freq;
        float fcr = hls::cosf(val);
        float fci = hls::sinf(val);
        
        float v0 = q_in[i];
        float v1 = q_in[i + 1];
        q_out[i] = v0 * fcr - v1 * fci;
        q_out[i + 1] = v0 * fci + v1 * fcr;
    }
}

void QKV_matmul_rotation(int pos, float q[dim], float k[kv_dim], float v[kv_dim], 
                int8_t xq_q[dim], float xq_s[dim/GS],
                int8_t *wq_weights, float *wq_scales,
                int8_t *wk_weights, float *wk_scales,
                int8_t *wv_weights, float *wv_scales
                ) {


    // TODO: Research workaround for xq_q/xq_s reuse to dataflow

    float q_prerope[dim];
    float k_prerope[kv_dim];
    #pragma HLS ARRAY_PARTITION variable=q_prerope type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=k_prerope type=cyclic factor=8

    matmul<dim, dim>(q_prerope, xq_q, xq_s, wq_weights, wq_scales);
    matmul<kv_dim, dim>(k_prerope, xq_q, xq_s, wk_weights, wk_scales);
    matmul<kv_dim, dim>(v, xq_q, xq_s, wv_weights, wv_scales);

    rope_rotation(q_prerope, k_prerope, q, k, pos);
}

// TODO: needs optimizing, caches are randomly accessed
void multihead_attention_with_cache(float q[dim], float k[kv_dim], float v[kv_dim], float *key_cache, float *value_cache, 
    float xb[dim], float att[n_heads * seq_len], int layer, int pos) {

    const int kv_cache_layer_offset = layer * seq_len * kv_dim;
    const int kv_cache_pos_offset = kv_cache_layer_offset + pos * kv_dim;
    const float inv_sqrt_head_size = 1.0f / hls::sqrtf((float)head_size);

    update_key:
    for (int i = 0; i < kv_dim; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768
        key_cache[kv_cache_pos_offset + i] = k[i];
    }
    
    update_value:
    for (int i = 0; i < kv_dim; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768
        value_cache[kv_cache_pos_offset + i] = v[i];
    }
    
    multihead_loop:
    for (int h = 0; h < n_heads; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12
        
        float *q_head = q + h * head_size;
        float *att_head = att + h * seq_len;
        
        // Compute attention scores for this head
        att_scores:
        for (int t = 0; t <= pos; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024

            int kv_head = h / (n_heads / n_kv_heads);
            // TODO: k_head non-sequential access to key cache, preload weights?
            float *k_head = &key_cache[kv_cache_layer_offset + t * kv_dim + kv_head * head_size];
            float score = 0.0f;

            for (int i = 0; i < head_size; i++) {
                #pragma HLS UNROLL factor=8

                score += q_head[i] * k_head[i];
            }
            att_head[t] = score * inv_sqrt_head_size;
        }
        
        // Softmax over attention scores
        softmax<seq_len>(att_head, pos + 1);
        
        // Weighted sum of values
        float *xb_head = xb + h * head_size;
        
        init_xb:
        for (int i = 0; i < head_size; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            xb_head[i] = 0.0f;
        }
        
        att_weighted_sum:
        for (int t = 0; t <= pos; t++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024            
            
            int kv_head = h / (n_heads / n_kv_heads);
            // TODO: v_head non-sequential access to value cache, preload weights?

            float *v_head = &value_cache[kv_cache_layer_offset + t * kv_dim + kv_head * head_size];
            float a = att_head[t];
            
            for (int i = 0; i < head_size; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64

                xb_head[i] += a * v_head[i];
            }
        }
    }
}

// FFN BLOCK HELPER ================================================================

void swiglu_activation(float hb[hidden_dim], float hb2[hidden_dim], float o[hidden_dim]) {

    
    swiglu_loop:
    for (int i = 0; i < hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=2048 max=2048

        float val = hb[i];
        val *= (1.0f / (1.0f + hls::expf(-val)));   // SiLU/Swish
        o[i] = val * hb2[i];                        // Gated
    }
}

void FFN_matmul(float o[hidden_dim], 
                int8_t xq_q[dim], float xq_s[dim/64],
                int8_t *w1_weights, float *w1_scales,
                int8_t *w3_weights, float *w3_scales ){

    // TODO: Research workaround for xq_q/xq_s reuse
    // #pragma HLS DATAFLOW

    float hb[hidden_dim];
    float hb2[hidden_dim];
    #pragma HLS ARRAY_PARTITION variable=hb2 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=hb type=cyclic factor=8


    matmul<hidden_dim, dim>(hb, xq_q, xq_s, w1_weights, w1_scales);
    matmul<hidden_dim, dim>(hb2, xq_q, xq_s, w3_weights, w3_scales);
    swiglu_activation(hb, hb2, o);
}


// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================

// Attention block: RMSNorm -> QKV projection -> RoPE -> Attention -> Output projection
void attention_block( 
    int layer, int pos,
    float x[dim],
    float x_out[dim],
    float att[n_heads * seq_len],
    int8_t *wq_weights, float *wq_scales,
    int8_t *wk_weights, float *wk_scales,
    int8_t *wv_weights, float *wv_scales,
    int8_t *wo_weights, float *wo_scales,
    float *rms_att_weight,
    float *key_cache, float *value_cache
) {
    #pragma HLS DATAFLOW

    const int dim_dim_offset = layer * dim * dim;
    const int dim_kv_offset = layer * dim * kv_dim;
    const int rms_offset = layer * dim;

    // Local buffers
    float xb_norm[dim]; 
    int8_t xq_q_proj[dim];
    float xq_s_proj[dim/GS];
    #pragma HLS ARRAY_PARTITION variable=xb_norm type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_q_proj type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_s_proj type=complete

    float q[dim];
    float k[kv_dim];
    float v[kv_dim];
    #pragma HLS ARRAY_PARTITION variable=q type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=k type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=v type=cyclic factor=8

    float xb_att[dim];
    int8_t xq_q_qkv[dim];
    float xq_s_qkv[dim/GS]; 
    #pragma HLS ARRAY_PARTITION variable=xb_att type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_q_qkv type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_s_qkv type=complete
    
    // Attention preprocessing - Consumes x, Produce xq_q_qkv & xq_s_qkv
    rmsnorm<dim>(xb_norm, x, &rms_att_weight[rms_offset]);
    quantize<dim>(xq_q_qkv, xq_s_qkv, xb_norm);
    
    // QKV projections - Consumes xq_q_qkv & xq_s_qkv, Produce q, k, v
    QKV_matmul_rotation(pos, q, k, v, xq_q_qkv, xq_s_qkv, 
                        &wq_weights[dim_dim_offset], &wq_scales[dim_dim_offset / GS],
                        &wk_weights[dim_kv_offset], &wk_scales[dim_kv_offset / GS],
                        &wv_weights[dim_kv_offset], &wv_scales[dim_kv_offset / GS]);
    
    
    // Multi-head attention & cache update - Consumes q, k, v, and kv caches, Produce xb_att
    multihead_attention_with_cache(q, k, v, key_cache, value_cache, xb_att, att, layer, pos);
    
    // Output projection - Consumes xb_att, Produces xb_proj
    quantize<dim>(xq_q_proj, xq_s_proj, xb_att);
    matmul<dim, dim>(x_out, xq_q_proj, xq_s_proj, 
                    &wo_weights[dim_dim_offset], &wo_scales[dim_dim_offset / GS]);
}

// FFN block: RMSNorm -> SwiGLU -> Down projection
void ffn_block(
    int layer,
    float x[dim], float x_out[dim],
    int8_t *w1_weights, float *w1_scales,
    int8_t *w2_weights, float *w2_scales,
    int8_t *w3_weights, float *w3_scales,
    float *rms_ffn_weight
) {
    #pragma HLS DATAFLOW
    
    const int dim_hidden_offset = layer * dim * hidden_dim;
    const int hidden_dim_offset = layer * hidden_dim * dim;
    const int rms_offset = layer * dim;

    // Local buffers
    float xb_preproc[dim];
    int8_t xq_q_ffn[dim];
    float xq_s_ffn[dim/GS]; 
    #pragma HLS ARRAY_PARTITION variable=xb_preproc type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_q_ffn type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_s_ffn type=complete

    float hb_proj[hidden_dim];
    int8_t hq_q_proj[hidden_dim];
    float hq_s_proj[hidden_dim/GS]; 
    #pragma HLS ARRAY_PARTITION variable=hb_proj type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=hq_q_proj type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=hq_s_proj type=complete

    // FFN preprocessing - Consumes x, Produces xq_q_ffn, xq_s_ffn
    rmsnorm<dim>(xb_preproc, x, &rms_ffn_weight[rms_offset]);
    quantize<dim>(xq_q_ffn, xq_s_ffn, xb_preproc);
    
    // FFN forward (SwiGLU) - Consumes xq_q_ffn, xq_s_ffn, Produces hb_proj
    FFN_matmul(hb_proj, xq_q_ffn, xq_s_ffn, 
                &w1_weights[dim_hidden_offset], &w1_scales[dim_hidden_offset / GS], 
                &w3_weights[dim_hidden_offset], &w3_scales[dim_hidden_offset / GS]);
    
    // Project back to dim - Consumes hb_proj, Produces x_out
    quantize<hidden_dim>(hq_q_proj, hq_s_proj, hb_proj);
    matmul<dim, hidden_dim>(x_out, hq_q_proj, hq_s_proj, &w2_weights[hidden_dim_offset], &w2_scales[hidden_dim_offset / GS]);
}

// Final classification: RMSNorm -> Linear projection to vocab
void final_classifier(
    float x[dim], float *out,
    float *rms_final_weight,
    int8_t *wcls_weights, float *wcls_scales
) {
    #pragma HLS DATAFLOW

    float xb[dim];
    int8_t xq_q[dim];    
    float xq_s[dim/GS];

    #pragma HLS ARRAY_PARTITION variable=xb type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_q type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=xq_s type=complete
    
    rmsnorm<dim>(xb, x, rms_final_weight);
    quantize<dim>(xq_q, xq_s, xb);
    matmul<vocab_size, dim>(out, xq_q, xq_s, wcls_weights, wcls_scales);
}

// ============================================================================
// TOP-LEVEL FORWARD FUNCTION
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
    // ============================================================================
    // AXI Interface Pragmas - Memory Mapping
    // ============================================================================

    // Embedding
    #pragma HLS INTERFACE m_axi port=token_embedding_table offset=slave depth=24576000 bundle=gmem0 max_read_burst_length=256 max_widen_bitwidth=512
    
    // Attention weights
    #pragma HLS INTERFACE m_axi port=wq_weights offset=slave depth=7077888 bundle=gmem1 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wq_scales offset=slave depth=110592 bundle=gmem2 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wk_weights offset=slave depth=7077888 bundle=gmem3 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wk_scales offset=slave depth=110592 bundle=gmem4 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wv_weights offset=slave depth=7077888 bundle=gmem5 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wv_scales offset=slave depth=110592 bundle=gmem6 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wo_weights offset=slave depth=7077888 bundle=gmem7 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wo_scales offset=slave depth=110592 bundle=gmem8 max_read_burst_length=256 max_widen_bitwidth=512
    
    // FFN weights
    #pragma HLS INTERFACE m_axi port=w1_weights offset=slave depth=18874368 bundle=gmem9 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w1_scales offset=slave depth=294912 bundle=gmem10 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w2_weights offset=slave depth=18874368 bundle=gmem11 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w2_scales offset=slave depth=294912 bundle=gmem12 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w3_weights offset=slave depth=18874368 bundle=gmem13 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w3_scales offset=slave depth=294912 bundle=gmem14 max_read_burst_length=256 max_widen_bitwidth=512
    
    // RMS norm weights
    #pragma HLS INTERFACE m_axi port=rms_att_weight offset=slave depth=9216 bundle=gmem15 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=rms_ffn_weight offset=slave depth=9216 bundle=gmem16 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=rms_final_weight offset=slave depth=768 bundle=gmem17 max_read_burst_length=256 max_widen_bitwidth=512
    
    // Classifier weights
    #pragma HLS INTERFACE m_axi port=wcls_weights offset=slave depth=24576000 bundle=gmem18 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wcls_scales offset=slave depth=384000 bundle=gmem19 max_read_burst_length=256 max_widen_bitwidth=512
    
    // KV cache (read/write)
    #pragma HLS INTERFACE m_axi port=key_cache offset=slave depth=9437184 bundle=gmem20 max_read_burst_length=256 max_write_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=value_cache offset=slave depth=9437184 bundle=gmem21 max_read_burst_length=256 max_write_burst_length=256 max_widen_bitwidth=512
    
    // Output (write only)
    #pragma HLS INTERFACE m_axi port=out offset=slave depth=32000 bundle=gmem22 max_write_burst_length=256
    
    // Control interface
    #pragma HLS INTERFACE s_axilite port=token
    #pragma HLS INTERFACE s_axilite port=pos
    #pragma HLS INTERFACE s_axilite port=return

    // Local buffers
    float x[dim];
    float x_out[dim];
    float att[n_heads * seq_len];

    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=x_out type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=att type=cyclic factor=n_heads

    // ============================================================================
    // Forward Pass Execution
    // ============================================================================

    // Load embedding
    load_embedding(token_embedding_table, x, token);
    
    // Process all (12) transformer layers
    main_forward_loop:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12
        // TODO: Dataflow entire layer transforms (if possible)
        
        // Attention block - Dataflow
        attention_block(l, pos, x, x_out, att,
                       wq_weights, wq_scales, wk_weights, wk_scales,
                       wv_weights, wv_scales, wo_weights, wo_scales,
                       rms_att_weight, key_cache, value_cache);
        
        residual_add<dim>(x, x_out);

        // FFN block - Dataflow
        ffn_block(l, x, x_out,
                 w1_weights, w1_scales, w2_weights, w2_scales,
                 w3_weights, w3_scales, rms_ffn_weight);

        residual_add<dim>(x, x_out);
    }
    
    // Final classifier - Dataflow
    final_classifier(x, out, rms_final_weight, wcls_weights, wcls_scales);
}