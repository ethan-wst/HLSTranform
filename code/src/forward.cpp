#include "forward.h"
#include "config.h"
#include <cstring>
#include <cmath>

// Main forward function with minimal interface pragmas
extern "C" void forward(
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer,
    int token, 
    int pos, 
    float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float *out
) {

    // Disable automatic inlining 
    #pragma HLS INLINE off

    // Interface pragmas
    #pragma HLS INTERFACE m_axi port=transformer offset=slave
    #pragma HLS INTERFACE m_axi port=key_cache offset=slave
    #pragma HLS INTERFACE m_axi port=value_cache offset=slave
    #pragma HLS INTERFACE m_axi port=out offset=slave
    // Control interface for scalars
    #pragma HLS INTERFACE s_axilite port=token
    #pragma HLS INTERFACE s_axilite port=pos
    #pragma HLS INTERFACE s_axilite port=return
    
    // Static arrays
    static float x[dim];                                                    // activation at current time stamp (dim)
    static float xb[dim];                                                   // same, but inside a residual branch (dim)
    static float xb2[dim];                                                  // an additional buffer just for convenience (dim)
    static float hb[hidden_dim];                                            // buffer for hidden dimension in the ffn (hidden_dim)
    static float hb2[hidden_dim];                                           // buffer for hidden dimension in the ffn (hidden_dim)
    static QuantizedTensor<dim> xq;                                         // quantized x (dim)
    static QuantizedTensor<hidden_dim> hq;                                  // quantized hb (hidden_dim)
    static float q[dim];                                                    // query (dim)
    static float k[(dim * n_kv_heads) / n_heads];                           // key (dim)
    static float v[(dim * n_kv_heads) / n_heads];                           // value (dim)
    static float att[n_heads * seq_len];                                    // buffer for scores/attention values (n_heads, seq_len)
    
    // Key constants
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;                    // dimension of key/value vectors
    constexpr int kv_mul = n_heads / n_kv_heads;                            // integer multiplier of the kv sharing in multiquery
    constexpr int head_size = dim / n_heads;                                // dimension of each attention head

    // Pre-compute reciprocals for frequent divisions
    static const float inv_head_size = 1.0f / float(head_size);
    static const float inv_sqrt_head_size = 1.0f / sqrtf(float(head_size));
    constexpr float inv_10000 = 1.0f / 10000.0f;
        
    // Access transformer weights
    auto w = &transformer->weights;
    
    // Copy the token embedding into x
    std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));
    
    main_forward_loop:
    for (int l = 0; l < n_layers; l++) {
        // Disable automatic loop optimizations
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off
        
        // Attention rmsnorm
        rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);
        
        // QKV matmuls for this position
        quantize<dim>(&xq, xb, GS);
        matmul<dim, dim, GS>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
        matmul<kv_dim, dim, GS>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
        matmul<kv_dim, dim, GS>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);
        
        // RoPE
        rotation1:
        for (int i = 0; i < kv_dim; i += 2) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true

            int head_dim = i % head_size;
            float freq = powf(inv_10000, head_dim * inv_head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
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
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true
            int head_dim = i % head_size;
            float freq = powf(inv_10000, head_dim * inv_head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            // Rotate only the query vector
            float v0 = q[i];
            float v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        
        // Save key,value at this time step (pos) to our kv cache
        int loff = l * seq_len * kv_dim;
        float *key_cache_row = key_cache + loff + pos * kv_dim;
        float *value_cache_row = value_cache + loff + pos * kv_dim;
        std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
        std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));
        
        multihead_attention:
        for (int h = 0; h < n_heads; h++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true

            // Get the query vector for this head
            const int q_offset = h * head_size;
            
            // Attention scores for this head
            const int att_offset = h * seq_len;
            
            // Iterate over all timesteps, including the current one
            iterate:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE off
                #pragma HLS UNROLL off=true

                // Get the key vector for this head and at this timestep
                const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                
                // Calculate the attention score as the dot product of q and k
                float score = 0.0f;
                attention_dot:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS PIPELINE off
                    #pragma HLS UNROLL off=true

                    score += q[i + q_offset] * key_cache[i + key_offset];
                }
                score *= inv_sqrt_head_size;  // Scale the score
                
                // Save the score to the attention buffer
                att[t + att_offset] = score;
            }
            
            // Softmax the scores to get attention weights, from 0..pos inclusively - FIXED: Added template parameter
            softmax<seq_len>(att + att_offset, pos + 1);
            
            // Weighted sum of the values, store back into xb
            const int xb_offset = h * head_size;
            memset(xb + xb_offset, 0, head_size * sizeof(float));
            
            acc:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE off
                #pragma HLS UNROLL off=true

                // Get the value vector for this head and at this timestep
                const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                
                // Get the attention weight for this timestep
                float a = att[t + att_offset];
                
                // Accumulate the weighted value into xb
                acc_inner:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS PIPELINE off
                    #pragma HLS UNROLL off=true

                    xb[i + xb_offset] += a * value_cache[i + v_offset];
                }
            }
        }

        // Final matmul to get the output of the attention
        quantize<dim>(&xq, xb, GS);
        matmul<dim, dim, GS>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);
        
        // Residual connection back into x
        residual:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true

            x[i] += xb2[i];
        }

        // FFN rmsnorm
        rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
        
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // First calculate self.w1(x) and self.w3(x)
        quantize<dim>(&xq, xb, GS);
        matmul<hidden_dim, dim, GS>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
        matmul<hidden_dim, dim, GS>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);
        
        float hb_out[hidden_dim];
        
        // SwiGLU activation: silu(x) = x * sigmoid(x)
        swi_glu:
        for (int i = 0; i < hidden_dim; i++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true

            float val = hb[i];

            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            float exp_neg_val = expf(-val);
            val *= (1.0f / (1.0f + exp_neg_val));

            // elementwise multiply with w3(x)
            val *= hb2[i];
            hb_out[i] = val;
        }
        
        std::memcpy(hb, hb_out, hidden_dim * sizeof(float));

        quantize<hidden_dim>(&hq, hb, GS);
        matmul<dim, hidden_dim, GS>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);
        
        residual2:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true
            x[i] += xb[i];
        }
    }
    
    rmsnorm<dim>(x, x, w->rms_final_weight);
    
    // Classifier into logits
    quantize<dim>(&xq, x, GS);
    matmul<vocab_size, dim, GS>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}

template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
    #pragma HLS INLINE off

    // Calculate sum of squares
    float ss = 0.0f;
    
    sum_of_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        float x_j = x[j];
        ss += x_j * x_j;
    }

    ss /= S;
    ss += 1e-5f;
    float inv_sqrt_ss = 1.0f / sqrtf(ss);

    norm_and_scale:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        o[j] = weight[j] * (inv_sqrt_ss * x[j]);
    }
}

template<int MAXSIZE>
void softmax(float *x, int size) {
    #pragma HLS INLINE off

    // Find max value (for numerical stability)
    float max_val = x[0];
    
    max:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        float x_i = x[i];
        if (x_i > max_val) {
            max_val = x_i;
        }
    }
    
    // Exp and sum
    float sum = 0.0f;
    
    exp_and_sum:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        float x_i = expf(x[i] - max_val);
        x[i] = x_i;
        sum += x_i;
    }

    // Normalize
    const float inv_sum = 1.0f / sum;
    
    norm:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        x[i] = x[i] * inv_sum;
    }
}

template<int D, int N, int GS>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
    #pragma HLS INLINE off

    // W (d,n) @ x (n,) -> xout (d,)
    // Quantized matrix multiplication
    outer_matmul:
    for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true

        float val = 0.0f;
        
        // Do the matmul in groups of GS
        inner_matmul:
        for (int j = 0; j <= N - GS; j += GS) {
            #pragma HLS PIPELINE off
            #pragma HLS UNROLL off=true

            int32_t ival = 0;
            
            // Inner product for this group
            grouped_dot:
            for (int k = 0; k < GS; k++) {
                #pragma HLS PIPELINE off
                #pragma HLS UNROLL off=true

                ival += ((int32_t)xq[j + k]) * ((int32_t)wq[i * N + j + k]);
            }
            
            // Scale and accumulate
            val += ((float)ival) * ws[i * N / GS + j / GS] * xs[j / GS];
        }
        xout[i] = val;
    }
}

// Explicit template instantiations for HLS
template void rmsnorm<dim>(float o[dim], float x[dim], float weight[dim]);
template void rmsnorm<hidden_dim>(float o[hidden_dim], float x[hidden_dim], float weight[hidden_dim]);
template void softmax<seq_len>(float *x, int size);
template void matmul<dim, dim, GS>(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);
template void matmul<hidden_dim, dim, GS>(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);
template void matmul<vocab_size, dim, GS>(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);
