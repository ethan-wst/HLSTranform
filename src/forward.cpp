// Modified forward.cpp with optimized interfaces and memory mapping for U50
#include "forward.h"
#include "config.h"
#include <cstring>
#include <cmath>

// Main forward function with U50-optimized interfaces
extern "C" void forward(
    Transformer *transformer,
    int token, 
    int pos, 
    float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float *out
) {
    // Interface pragmas optimized for U50 HBM access
    #pragma HLS INTERFACE m_axi port=transformer bundle=gmem0 offset=slave depth=1 max_read_burst_length=256 max_write_burst_length=64 latency=64
    #pragma HLS INTERFACE m_axi port=key_cache bundle=gmem1 offset=slave max_read_burst_length=256 max_write_burst_length=256 latency=64
    #pragma HLS INTERFACE m_axi port=value_cache bundle=gmem2 offset=slave max_read_burst_length=256 max_write_burst_length=256 latency=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem3 offset=slave max_write_burst_length=64 latency=64
    
    // Control interface for scalars
    #pragma HLS INTERFACE s_axilite port=token bundle=control
    #pragma HLS INTERFACE s_axilite port=pos bundle=control  
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Local arrays with memory partitioning for optimal access patterns
    static float x[dim];
    static float xb[dim]; 
    static float q[dim];
    static float k[(dim * n_kv_heads) / n_heads];
    static float v[(dim * n_kv_heads) / n_heads];
    static float att[n_heads * seq_len];
    static float logits[vocab_size];
    
    // Memory partitioning for parallel access
    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=xb type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=q type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=k type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=v type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=att type=cyclic factor=16
    
    // Token embedding lookup
    float* content_row = transformer->token_embedding_table + token * dim;
    
    // Copy token embedding to x with burst optimization
    memcpy_token:
    for (int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768 avg=768
        x[i] = content_row[i];
    }
    
    // Forward through transformer layers
    transformer_layers:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12 avg=12
        
        // Attention
        rmsnorm<dim>(xb, x, transformer->rms_att_weight + l * dim);
        
        // QKV projections with optimized matmul calls
        matmul<dim, 1, GS>(q, transformer->wq + l * dim * dim, xb);
        matmul<(dim * n_kv_heads) / n_heads, 1, GS>(k, transformer->wk + l * dim * ((dim * n_kv_heads) / n_heads), xb);
        matmul<(dim * n_kv_heads) / n_heads, 1, GS>(v, transformer->wv + l * dim * ((dim * n_kv_heads) / n_heads), xb);
        
        // RoPE relative positional encoding
        rope_encoding:
        for (int i = 0; i < dim; i += 2) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=384 max=384 avg=384
            
            int head_dim = dim / n_heads;
            float freq = 1.0f / powf(10000.0f, (float)(i % head_dim) / (float)head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            int rotn = (i < dim / 2) ? 0 : 1;
            if (rotn == 0) {
                float v0 = q[i];
                float v1 = q[i+1];
                q[i] = v0 * fcr - v1 * fci;
                q[i+1] = v0 * fci + v1 * fcr;
                
                if (i < (dim * n_kv_heads) / n_heads) {
                    v0 = k[i];
                    v1 = k[i+1];
                    k[i] = v0 * fcr - v1 * fci;
                    k[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }
        
        // Store key and value in cache
        int loff = l * seq_len * ((dim * n_kv_heads) / n_heads);
        float* key_cache_row = key_cache + loff + pos * ((dim * n_kv_heads) / n_heads);
        float* value_cache_row = value_cache + loff + pos * ((dim * n_kv_heads) / n_heads);
        
        store_kv_cache:
        for (int i = 0; i < (dim * n_kv_heads) / n_heads; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
            key_cache_row[i] = k[i];
            value_cache_row[i] = v[i];
        }
        
        // Multihead attention
        int head_size = dim / n_heads;
        multihead_attention:
        for (int h = 0; h < n_heads; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=12 max=12 avg=12
            
            float* q_head = q + h * head_size;
            
            // Calculate attention scores
            attention_scores:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=2
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 avg=512
                
                float* k_head = key_cache + loff + t * ((dim * n_kv_heads) / n_heads) + (h / (n_heads / n_kv_heads)) * head_size;
                
                float score = 0.0f;
                dot_product:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS UNROLL factor=8
                    score += q_head[i] * k_head[i];
                }
                score /= sqrtf((float)head_size);
                att[h * seq_len + t] = score;
            }
            
            // Softmax
            float max_val = att[h * seq_len];
            find_max_att:
            for (int t = 1; t <= pos; t++) {
                #pragma HLS PIPELINE II=1
                max_val = (att[h * seq_len + t] > max_val) ? att[h * seq_len + t] : max_val;
            }
            
            float sum = 0.0f;
            softmax_exp:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 avg=512
                att[h * seq_len + t] = expf(att[h * seq_len + t] - max_val);
                sum += att[h * seq_len + t];
            }
            
            softmax_norm:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 avg=512
                att[h * seq_len + t] /= sum;
            }
            
            // Weighted sum of values
            float* xb_head = xb + h * head_size;
            reset_head:
            for (int i = 0; i < head_size; i++) {
                #pragma HLS UNROLL factor=8
                xb_head[i] = 0.0f;
            }
            
            weighted_sum:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE II=2
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1024 avg=512
                
                float* v_head = value_cache + loff + t * ((dim * n_kv_heads) / n_heads) + (h / (n_heads / n_kv_heads)) * head_size;
                float a = att[h * seq_len + t];
                
                accumulate_values:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS UNROLL factor=8
                    xb_head[i] += a * v_head[i];
                }
            }
        }
        
        // Final attention projection
        matmul<dim, 1, GS>(xb, transformer->wo + l * dim * dim, xb);
        
        // Residual connection
        residual_att:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768 avg=768
            x[i] += xb[i];
        }
        
        // FFN
        rmsnorm<dim>(xb, x, transformer->rms_ffn_weight + l * dim);
        
        // FFN projections
        matmul<hidden_dim, 1, GS>(xb, transformer->w1 + l * dim * hidden_dim, xb);
        matmul<hidden_dim, 1, GS>(x, transformer->w3 + l * dim * hidden_dim, xb);
        
        // SwiGLU activation
        swiglu_activation:
        for (int i = 0; i < hidden_dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=2048 max=2048 avg=2048
            float silu = xb[i] / (1.0f + expf(-xb[i]));
            xb[i] = silu * x[i];
        }
        
        matmul<dim, 1, GS>(xb, transformer->w2 + l * hidden_dim * dim, xb);
        
        // Residual connection
        residual_ffn:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768 avg=768
            x[i] += xb[i];
        }
    }
    
    // Final rmsnorm
    rmsnorm<dim>(x, x, transformer->rms_final_weight);
    
    // Classifier
    matmul<vocab_size, 1, GS>(logits, transformer->wcls, x);
    
    // Copy output
    copy_output:
    for (int i = 0; i < vocab_size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=32000 max=32000 avg=32000
        out[i] = logits[i];
    }
}

// Neural network building blocks with HLS optimizations
template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
    #pragma HLS INLINE off
    
    float x_buff[S];
    float weight_buff[S]; 
    float out_buff[S];
    
    #pragma HLS ARRAY_PARTITION variable=x_buff type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=weight_buff type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=out_buff type=cyclic factor=16
    
    // Calculate sum of squares
    float ss = 0.0f;
    sum_of_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=2048 avg=768
        float x_j = x[j];
        ss += x_j * x_j;
    }
    
    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    // Normalize and scale
    normalize:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=2048 avg=768
        o[j] = weight[j] * (ss * x[j]);
    }
}

template<int D, int N, int GS>
void matmul(float *xout, int8_t *w, float *x) {
    #pragma HLS INLINE off
    
    matmul_outer:
    for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE II=8
        #pragma HLS LOOP_TRIPCOUNT min=768 max=32000 avg=768
        
        float val = 0.0f;
        
        matmul_inner: 
        for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL factor=8
            val += (float)w[i * N + j] * x[j];
        }
        xout[i] = val;
    }
}