// Modified forward.h with optimized interfaces for U50 HBM
#include "typedefs.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>

// Forward declaration with optimized interface for U50
extern "C" void forward(
    Transformer *transformer,
    int token, 
    int pos, 
    float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
    float *out
);

template<int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=16
    
    dequant_loop:
    for (int i = 0; i < S; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=768 max=2048 avg=768
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

template<int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=16
    
    constexpr int num_groups = S / 64;
    constexpr float Q_MAX = 127.0f;
    
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];
    
    #pragma HLS ARRAY_PARTITION variable=quantized_buffer type=cyclic factor=64
    #pragma HLS ARRAY_PARTITION variable=scale_buffer type=cyclic factor=16
    
    main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=12 max=32 avg=12
        
        float wmax = 0.0;
        int base_idx = group * GS;
        
        // Find max value in group
        find_max:
        for (int i = 0; i < GS; i++) {
            #pragma HLS UNROLL factor=8
            float abs_val = (x[base_idx + i] >= 0) ? x[base_idx + i] : -x[base_idx + i];
            wmax = (abs_val > wmax) ? abs_val : wmax;
        }
        
        float scale = wmax / Q_MAX;
        scale_buffer[group] = scale;
        
        // Quantize values in group
        quantize_group:
        for (int i = 0; i < GS; i++) {
            #pragma HLS UNROLL factor=8
            float val = x[base_idx + i] / scale;
            quantized_buffer[base_idx + i] = (int8_t)(val + 0.5f);
        }
    }
    
    // Copy results back
    copy_results:
    for (int i = 0; i < S; i++) {
        #pragma HLS PIPELINE II=1
        qx->q[i] = quantized_buffer[i];
    }
    
    copy_scales:
    for (int g = 0; g < num_groups; g++) {
        #pragma HLS PIPELINE II=1
        qx->s[g] = scale_buffer[g];
    }
}