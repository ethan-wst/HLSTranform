#include "typedefs.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>

// Use the config values to create concrete types
typedef Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> Transformer;

// Forward declaration with concrete types
extern "C" void forward(
   Transformer *transformer,
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

template<int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    constexpr int num_groups = S / 64;
    constexpr float Q_MAX = 127.0f;
    
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];
    
    main_loop:
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        int base_idx = group * GS;
        
        // Find max value in group
        find_max:
        for (int i = 0; i < GS; i++) {
            float abs_val = (x[base_idx + i] >= 0) ? x[base_idx + i] : -x[base_idx + i];
            wmax = (abs_val > wmax) ? abs_val : wmax;
        }
        
        float scale = wmax / Q_MAX;
        scale_buffer[group] = scale;
        
        // Quantize values in group
        quantize_group:
        for (int i = 0; i < GS; i++) {
            float val = x[base_idx + i] / scale;
            quantized_buffer[base_idx + i] = (int8_t)(val + 0.5f);
        }
    }
    
    // Copy results back
    copy_results:
    for (int i = 0; i < S; i++) {
        qx->q[i] = quantized_buffer[i];
    }
    
    copy_scales:
    for (int g = 0; g < num_groups; g++) {
        qx->s[g] = scale_buffer[g];
    }
}