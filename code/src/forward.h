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

// Function declarations
template<int S>
void rmsnorm(float o[S], float x[S], float weight[S]);

template<int D, int N, int GS>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws);

template<int MAXSIZE>
void softmax(float *x, int size);

// Template implementations
template<int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    dequant_loop:
    for (int i = 0; i < S; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

template<int S, int GS>
void quantize(QuantizedTensor<S> *qx, float x[S]) {
    #pragma HLS INLINE off

    constexpr int num_groups = S / GS;
    constexpr float inv_Q_MAX = 1 / 127.0f;

    main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS PIPELINE

        int base_idx = group * GS;
        float wmax = 0.0f;

        // Calculate the max absolute value in the current group
        find_max:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE off

            float abs_val = hls::fabs(x[base_idx + i]);
            if (abs_val > wmax) {
                wmax = abs_val;
            }
        }

        // Calculate and write scalling factor
        float scale = wmax * inv_Q_MAX;
        float inv_scale = 1 / scale;
        qx->s[group] = scale;
        
        // Quantize values in group
        quantize_group:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE
            #pragma HLS UNROLL factor=8 skip_exit_check

            float val = x[base_idx + i] * inv_scale;
            qx->q[base_idx + i] = (int8_t)(val + 0.5f);
        }
    }
}