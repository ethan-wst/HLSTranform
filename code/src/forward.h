#include "typedefs.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>



// Forward declaration with concrete types
extern "C" void forward(
   Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer,
    int token, 
    int pos, 
    float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float *out
);



// Function Declarations ================================================================

template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {

    float w_buffer[S];
    #pragma HLS ARRAY_PARTITION variable=w_buffer type=cyclic factor=16
    rms_buffer:
    for (int i = 0; i < S; i ++) {
        #pragma HLS PIPELINE II=1
        w_buffer[i] = weight[i];
    }

    // Calculate sum of squares
    float ss = 0.0f;
    
    sum_of_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=16
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        ss += x[j] * x[j];
    }

    ss /= S;
    ss += 1e-5f;
    float inv_sqrt_ss = 1.0f / hls::sqrtf(ss);

    norm_and_scale:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=16
        #pragma HLS LOOP_TRIPCOUNT min=768 max=768

        o[j] = w_buffer[j] * (inv_sqrt_ss * x[j]);
    }
}

template<int S>
void quantize(QuantizedTensor<S> *qx, float x[S]) {

    constexpr int num_groups = S / GS;
    constexpr float inv_Q_MAX = 1 / 127.0f;

    main_loop:
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        int base_idx = group * GS;
        
        // Find max value in group
        find_max:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            float abs_val = (x[base_idx + i] >= 0) ? x[base_idx + i] : -x[base_idx + i];
            wmax = (abs_val > wmax) ? abs_val : wmax;
        }

        float scale = wmax * inv_Q_MAX;
        float inv_scale = 1 / scale;
        qx->s[group] = scale;
        
        // Quantize values in group
        quantize_group:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=4
            float val = x[base_idx + i] * inv_scale;
             qx->q[base_idx + i] = (int8_t)(val + 0.5f);
        }
    }
}

template<int D, int N>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {

    int8_t wq_buffer[N];
    float ws_buffer[N/GS];

    #pragma HLS ARRAY_PARTITION variable=wq_buffer type=cyclic factor=GS
    #pragma HLS ARRAY_PARTITION variable=ws_buffer type=complete

    outer_matmul:
    for (int i = 0; i < D; i++) {
        #pragma HLS PIPELINE off

        load_wq:
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            wq_buffer[j] = wq[(i * N) + j];
        }

        load_ws:
        for (int j = 0; j < N/GS; j++) {
            #pragma HLS PIPELINE II=1
            ws_buffer[j] = ws[(i * N) + j];
        }

        float val = 0.0f;

        inner_matmul:
        for (int g = 0; g < N/GS; g++) {
            #pragma HLS PIPELINE II=1

            int32_t ival = 0;
            
            grouped_dot:
            for (int k = 0; k < GS; k++) {
                #pragma HLS UNROLL

                ival += ((int32_t)xq[g * GS + k]) * ((int32_t)wq_buffer[g * GS + k]);
            }
            
            // Scale and accumulate
            val += ((float)ival) * ws_buffer[g] * xs[g];
        }
        xout[i] = val;
    }
}

template<int MAXSIZE>
void softmax(float *x, int size) {

    // Find max value (for numerical stability)
    float max_val = x[0];
    
    max:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE

        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Exp and sum
    float sum = 0.0f;
    
    exp_and_sum:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE

        float x_i = hls::expf(x[i] - max_val);
        x[i] = x_i;
        sum += x_i;
    }

    // Normalize
    const float inv_sum = 1.0f / sum;
    
    norm:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAXSIZE

        x[i] *= inv_sum;
    }
}


