#include "forward.h"
#include "config.h"
#include <cstring>
#include <cmath>
#include <hls_math.h>


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
    #pragma HLS INTERFACE m_axi port=transformer offset=slave bundle=gem0 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=key_cache offset=slave bundle=gem1 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=value_cache offset=slave bundle=gem2 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gem3 max_write_burst_length=256
    // Control interface for scalars
    #pragma HLS INTERFACE s_axilite port=token
    #pragma HLS INTERFACE s_axilite port=pos
    #pragma HLS INTERFACE s_axilite port=return
    
    // Static arrays
    static float x[dim] = {0};                                                    // activation at current time stamp (dim)
    static float xb[dim] = {0};                                                   // same, but inside a residual branch (dim)
    static float xb2[dim] = {0};                                                  // an additional buffer just for convenience (dim)
    static float hb[hidden_dim] = {0};                                            // buffer for hidden dimension in the ffn (hidden_dim)
    static float hb2[hidden_dim] = {0};                                           // buffer for hidden dimension in the ffn (hidden_dim)
    static QuantizedTensor<dim> xq;                                         // quantized x (dim)
    static QuantizedTensor<hidden_dim> hq;                                  // quantized hb (hidden_dim)
    static float q[dim] = {0};                                                    // query (dim)
    static float k[(dim * n_kv_heads) / n_heads] = {0};                           // key (dim)
    static float v[(dim * n_kv_heads) / n_heads] = {0};                           // value (dim)
    static float att[n_heads * seq_len] = {0};                                    // buffer for scores/attention values (n_heads, seq_len)

    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=xb type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=xb2 type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=hb type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=hb2 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=xq.q type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=xq.s type=complete
    #pragma HLS ARRAY_PARTITION variable=hq.q type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=hq.s type=complete
    #pragma HLS ARRAY_PARTITION variable=q type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=k type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=v type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=att type=cyclic factor=16
    
    
    // Key constants
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    constexpr int kv_mul = n_heads / n_kv_heads;
    constexpr int head_size = dim / n_heads;

    // Pre-compute reciprocals for frequent divisions
    static const float inv_head_size = 1.0f / float(head_size);
    static const float inv_sqrt_head_size = 1.0f / hls::sqrtf(float(head_size));
    constexpr float inv_10000 = 1.0f / 10000.0f;
        
    // Access transformer weights
    auto w = &transformer->weights;

    // Copy from embedding table
    for (int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        x[i] = w->token_embedding_table[token * dim + i];
    }
    
    main_forward_loop:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=12 max=12
        
        // Consumes x, produces xb
        rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);
        
        // Consumes xb, produces xq
        quantize<dim>(&xq, xb);
        // Consumes xq, produces q, k, v
        matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
        matmul<kv_dim, dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
        matmul<kv_dim, dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);
        
        // RoPE
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
        
        // Save key,value at this time step (pos) to our kv cache
        int loff = l * seq_len * kv_dim;
        float *key_cache_row = key_cache + loff + pos * kv_dim;
        float *value_cache_row = value_cache + loff + pos * kv_dim;
        std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
        std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

        // // Write key cache with pipelined burst
        // write_key:
        // for (int i = 0; i < kv_dim; i++) {
        //     #pragma HLS PIPELINE II=1
        //     #pragma HLS LOOP_TRIPCOUNT min=768 max=768
        //     key_cache_row[i] = k[i];
        // }
        
        // // Write value cache with pipelined burst
        // write_value:
        // for (int i = 0; i < kv_dim; i++) {
        //     #pragma HLS PIPELINE II=1
        //     #pragma HLS LOOP_TRIPCOUNT min=768 max=768
        //     value_cache_row[i] = v[i];
        // }
        
        multihead_attention:
        for (int h = 0; h < n_heads; h++) {
            #pragma HLS PIPELINE off
            #pragma HLS LOOP_TRIPCOUNT min=12 max=12

            const int q_offset = h * head_size;
            const int att_offset = h * seq_len;
            
            // Iterate over all timesteps, including the current one
            iterate:
            for (int t = 0; t <= pos; t++) {

                const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;

                // float key_buffer[head_size];
                // #pragma HLS ARRAY_PARTITION variable=key_buffer type=cyclic factor=8
                
                // load_key:
                // for (int i = 0; i < head_size; i++) {
                //     #pragma HLS PIPELINE II=1
                //     #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                //     key_buffer[i] = key_cache[i + key_offset];
                // }
                
                // Calculate the attention score as the dot product of q and k
                float score = 0.0f;
                attention_dot:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS UNROLL factor=8

                    // score += q[i + q_offset] * key_buffer[i];
                    score += q[i + q_offset] * key_cache[i + key_offset];
                }

                score *= inv_sqrt_head_size;
                att[t + att_offset] = score;
            }
            
            // Softmax the scores to get attention weights, from 0..pos inclusively - FIXED: Added template parameter
            softmax<seq_len>(att + att_offset, pos + 1);
            
            // Weighted sum of the values, store back into xb
            const int xb_offset = h * head_size;
            init_xb:
            for (int i = 0; i < head_size; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                xb[i + xb_offset] = 0.0f;
            }
            
            acc:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS PIPELINE off

                // Get the value vector for this head and at this timestep
                const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                
                // Get the attention weight for this timestep
                float a = att[t + att_offset];

                // Buffer value vector for this timestep
                // float value_buffer[head_size];
                // #pragma HLS ARRAY_PARTITION variable=value_buffer type=cyclic factor=8
                
                // Load value vector with burst
                // load_value:
                // for (int i = 0; i < head_size; i++) {
                //     #pragma HLS PIPELINE II=1
                //     #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                //     value_buffer[i] = value_cache[i + v_offset];
                // }
                
                // Accumulate the weighted value into xb
                acc_inner:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS UNROLL factor=8
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64

                    //xb[i + xb_offset] += a * value_buffer[i];
                    xb[i + xb_offset] += a * value_cache[i + v_offset];

                }
            }
        }

        // Final matmul to get the output of the attention
        quantize<dim>(&xq, xb);
        matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);
        
        // Residual connection back into x
        residual:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768


            x[i] += xb2[i];
        }

        // FFN rmsnorm
        rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
        
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // First calculate self.w1(x) and self.w3(x)
        quantize<dim>(&xq, xb);
        matmul<hidden_dim, dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
        matmul<hidden_dim, dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);
                
        // SwiGLU activation: silu(x) = x * sigmoid(x)
        swi_glu:
        for (int i = 0; i < hidden_dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            #pragma HLS LOOP_TRIPCOUNT min=2048 max=2048


            float val = hb[i];

            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + hls::expf(-val)));

            // elementwise multiply with w3(x)
            val *= hb2[i];
            hb[i] = val;
        }
        
        quantize<hidden_dim>(&hq, hb);
        matmul<dim, hidden_dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);
        
        residual2:
        for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            #pragma HLS LOOP_TRIPCOUNT min=768 max=768

            x[i] += xb[i];
        }
    }
    
    rmsnorm<dim>(x, x, w->rms_final_weight);
    
    // Classifier into logits
    quantize<dim>(&xq, x);
    matmul<vocab_size, dim>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}
