#include "forward.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <hls_math.h>
#include <ap_int.h>
#include <hls_stream.h>

// Top-level forward function with flattened weight interface
// All weights passed as individual pointers mapped to separate HBM banks

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

    // Embedding
    #pragma HLS INTERFACE m_axi port=token_embedding_table offset=slave depth=24576000 \
    bundle=gmem0 max_read_burst_length=256 num_read_outstanding=8 \
    max_widen_bitwidth=512

    // Attention
    #pragma HLS INTERFACE m_axi port=wq_weights offset=slave depth=7077888 \
    bundle=gmem1 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wq_scales offset=slave depth=110592 \
    bundle=gmem2 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=wk_weights offset=slave depth=7077888 \
    bundle=gmem3 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wk_scales offset=slave depth=110592 \
    bundle=gmem4 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=wv_weights offset=slave depth=7077888 \
    bundle=gmem5 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wv_scales offset=slave depth=110592 \
    bundle=gmem6 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=wo_weights offset=slave depth=7077888 \
    bundle=gmem7 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wo_scales offset=slave depth=110592 \
    bundle=gmem8 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    // FFN
    #pragma HLS INTERFACE m_axi port=w1_weights offset=slave depth=18874368 \
    bundle=gmem9 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w1_scales offset=slave depth=294912 \
    bundle=gmem10 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=w2_weights offset=slave depth=18874368 \
    bundle=gmem11 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w2_scales offset=slave depth=294912 \
    bundle=gmem12 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=w3_weights offset=slave depth=18874368 \
    bundle=gmem13 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=w3_scales offset=slave depth=294912 \
    bundle=gmem14 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    // RMS Norm
    #pragma HLS INTERFACE m_axi port=rms_att_weight offset=slave depth=9216 \
    bundle=gmem15 max_read_burst_length=64 num_read_outstanding=4 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=rms_ffn_weight offset=slave depth=9216 \
    bundle=gmem16 max_read_burst_length=64 num_read_outstanding=4 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=rms_final_weight offset=slave depth=768 \
    bundle=gmem17 max_read_burst_length=64 num_read_outstanding=4 \
    max_widen_bitwidth=512

    // Classifier
    #pragma HLS INTERFACE m_axi port=wcls_weights offset=slave depth=24576000 \
    bundle=gmem18 max_read_burst_length=256 num_read_outstanding=32 \
    max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=wcls_scales offset=slave depth=384000 \
    bundle=gmem19 max_read_burst_length=256 num_read_outstanding=16 \
    max_widen_bitwidth=512

    // KV Cache
    #pragma HLS INTERFACE m_axi port=key_cache offset=slave depth=9437184 \
    bundle=gmem20 \
    max_read_burst_length=64 max_write_burst_length=256 \
    num_read_outstanding=64 num_write_outstanding=8 \
    max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi port=value_cache offset=slave depth=9437184 \
    bundle=gmem21 \
    max_read_burst_length=64 max_write_burst_length=256 \
    num_read_outstanding=64 num_write_outstanding=8 \
    max_widen_bitwidth=512

    // Output
    #pragma HLS INTERFACE m_axi port=out offset=slave depth=32000 \
    bundle=gmem22 max_write_burst_length=256 num_write_outstanding=8 \
    max_widen_bitwidth=512

    // Control
    #pragma HLS INTERFACE s_axilite port=token
    #pragma HLS INTERFACE s_axilite port=pos
    #pragma HLS INTERFACE s_axilite port=return


    // ====================== LOCAL ARRAYS ========================             // dimension of each attention head

    // Main activation buffers
    float x[dim];                    // Current activation
    float xb[dim];                   // Intermediate buffer 1
    float xb2[dim];                  // Intermediate buffer 2
    float hb[hidden_dim];            // Hidden layer buffer
    float hb2[hidden_dim];           // Hidden layer buffer 2
    float q[dim];                    // Query
    float k[kv_dim];                 // Key
    float v[kv_dim];                 // Value
    float att[n_heads * seq_len];    // Attention scores     
    
    // Quantized tensors
    int8_t xq[dim];
    float xq_s[dim/GS];

    int8_t hq[hidden_dim];
    float hq_s[hidden_dim/GS];
    
    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=xb type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=xb2 type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=hb type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=hb2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=q type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=k type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=v type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=att type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=xq type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=xq_s type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=hq type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=hq_s type=cyclic factor=4

    // ================= FORWARD PASS PREPERATION =================

    // Key constants
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    constexpr int kv_mul = n_heads / n_kv_heads;
    constexpr int head_size = dim / n_heads;


    // Pre-compute reciprocals for frequent divisions
    static const float inv_head_size = 1.0f / float(head_size);
    static const float inv_sqrt_head_size = 1.0f / hls::sqrtf(float(head_size));
    constexpr float inv_10000 = 1.0f / 10000.0f;

    load_embedding:
    for (int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        x[i] = token_embedding_table[token * dim + i];
    }

    // ================= FORWARD PASS COMPUTATION =================
    
    main_forward_loop:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12

        // Calculate layer-specific offsets for weight access
        const int dim_dim_offset = l * dim * dim;
        const int dim_kv_offset = l * dim * kv_dim;
        const int dim_hidden_offset = l * dim * hidden_dim;
        const int hidden_dim_offset = l * hidden_dim * dim;
        const int rms_offset = l * dim;
        const int kv_cache_offset = l * seq_len * kv_dim;
        
        // ===================== ATTENTION BLOCK =====================

        rmsnorm<dim>(xb, x, &rms_att_weight[rms_offset]);
        
        quantize<dim>(xq, xq_s, xb);
        
        matmul<dim, dim>(q, xq, xq_s, &wq_weights[dim_dim_offset], &wq_scales[dim_dim_offset / GS]);
        matmul<kv_dim, dim>(k, xq, xq_s, &wk_weights[dim_kv_offset], &wk_scales[dim_kv_offset / GS]);
        matmul<kv_dim, dim>(v, xq, xq_s, &wv_weights[dim_kv_offset], &wv_scales[dim_kv_offset / GS]);
        
        rotation1:
        for (int i = 0; i < kv_dim; i += 2) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=384 max=384

            int head_dim = i % head_size;
            float freq = hls::powf(inv_10000, head_dim * inv_head_size);
            float val = pos * freq;
            float fcr = hls::cosf(val);
            float fci = hls::sinf(val);
            
            // Rotate the query vector
            float v0_q = q[i];
            float v1_q = q[i + 1];
            q[i] = v0_q * fcr - v1_q * fci;
            q[i + 1] = v0_q * fci + v1_q * fcr;
            
            // Rotate the key vector
            float v0_k = k[i];
            float v1_k = k[i + 1];
            k[i] = v0_k * fcr - v1_k * fci;
            k[i + 1] = v0_k * fci + v1_k * fcr;
        }
        
        rotation2:
        // Rotation for only the query vector (i >= kv_dim)
        for (int i = kv_dim; i < dim; i += 2) {
            #pragma HLS PIPELINE II=1

            int head_dim = i % head_size;
            float freq = hls::powf(inv_10000, head_dim * inv_head_size);
            float val = pos * freq;
            float fcr = hls::cosf(val);
            float fci = hls::sinf(val);
            
            // Rotate only the query vector
            float v0 = q[i];
            float v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        
        int kv_cache_pos_offset = kv_cache_offset + pos * kv_dim;
        
        update_kv_k:
        for (int i = 0; i < kv_dim; i++) {
            #pragma HLS PIPELINE II=1
            key_cache[kv_cache_pos_offset + i] = k[i];
        }
        
        update_kv_v:
        for (int i = 0; i < kv_dim; i++) {
            #pragma HLS PIPELINE II=1
            value_cache[kv_cache_pos_offset + i] = v[i];
        }
        

        multihead_attention:
        for (int h = 0; h < n_heads; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=12 max=12

            float *q_head = q + h * head_size;
            float *att_head = att + h * seq_len;
            
            // Compute attention scores for this head
            att_scores:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=1

                int kv_head = h / (n_heads / n_kv_heads);
                float *k_head = &key_cache[kv_cache_offset + t * kv_dim + kv_head * head_size];
                
                float score = 0.0f;
                attention_dot:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64


                    score += q_head[i] * k_head[i];
                }

                score *= inv_sqrt_head_size;
                att_head[t] = score;
            }
            
            // Softmax over attention scores            
            softmax<seq_len>(att_head, pos + 1);
            
            // Weighted sum of the values
            float *xb_head = xb + h * head_size;

            init_xb:
            for (int i = 0; i < head_size; i++) {
                #pragma HLS UNROLL
                xb_head[i] = 0.0f;
            }

            att_weighted_sum:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=1

                int kv_head = h / (n_heads / n_kv_heads);
                float *v_head = &value_cache[kv_cache_offset + t * kv_dim + kv_head * head_size];
                float a = att_head[t];
                
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS UNROLL factor=8
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64

                    xb_head[i] += a * v_head[i];
                }
            }
        }

        quantize<dim>(xq, xq_s, xb);
        matmul<dim, dim>(xb2, xq, xq_s, &wo_weights[dim_dim_offset], &wo_scales[dim_dim_offset / GS]);
        
        residual_att:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768

            x[i] += xb2[i];
        }

        // ===================== FFN BLOCK =====================

        rmsnorm<dim>(xb, x, &rms_ffn_weight[rms_offset]);

        quantize<dim>(xq, xq_s, xb);

        matmul<hidden_dim, dim>(hb, xq, xq_s, &w1_weights[dim_hidden_offset], &w1_scales[dim_hidden_offset / GS]);
        matmul<hidden_dim, dim>(hb2, xq, xq_s, &w3_weights[dim_hidden_offset], &w3_scales[dim_hidden_offset / GS]);
        
        // SwiGLU activation: silu(x) = x * sigmoid(x)
        swi_glu:
        for (int i = 0; i < hidden_dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            #pragma HLS LOOP_TRIPCOUNT min=2048 max=2048

            float val = hb[i];
            val *= (1.0f / (1.0f + hls::expf(-val)));
            val *= hb2[i];
            hb[i] = val;
        }
        
        quantize<hidden_dim>(hq, hq_s, hb);
        matmul<dim, hidden_dim>(xb, hq, hq_s, &w2_weights[hidden_dim_offset], &w2_scales[hidden_dim_offset / GS]);
        
        // Step 11: Residual connection (FFN)
        residual_ffn:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768

            x[i] += xb[i];
        }
    }

    // ==================== FINAL LAYER ======================

    rmsnorm<dim>(xb, x, rms_final_weight);

    // Classifier
    quantize<dim>(xq, xq_s, xb);
    matmul<vocab_size, dim>(out, xq, xq_s, wcls_weights, wcls_scales);
}