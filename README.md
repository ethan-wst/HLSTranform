# HLSTransform:  FPGA-Accelerated Llama 2 Transformer

Modified and optimized version of HLSTransform code for deployment to Xilinx Alveo U50 FPGA accelerator.

## Overview

This repository contains a High-Level Synthesis (HLS) implementation of a Llama 2 transformer model, specifically optimized for deployment on the **Xilinx Alveo U50** FPGA platform. The implementation leverages Vitis HLS to synthesize C++ code into hardware accelerators, enabling efficient inference with optimized memory bandwidth utilization across multiple HBM (High Bandwidth Memory) banks.

### Key Features

- **FPGA-Accelerated Inference**: Hardware implementation of Llama 2 forward pass using Xilinx Vitis HLS
- **Multi-Bank HBM Architecture**: Flattened weight interface mapped to up to 23 separate HBM banks for parallel memory access
- **Quantization Support**: INT8 weight quantization with per-group scaling (group size = 64) to reduce memory footprint
- **Optimized Memory Access**: Burst-optimized sequential array access with independent bandwidth optimization per weight type
- **Target Platform**: Xilinx Alveo U50 (xcu50-fsvh2104-2-e) @ 275MHz

## Architecture

### Model Configuration

The current implementation supports the following Llama 2 configuration (configurable in `code/src/config.h`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 768 | Transformer embedding dimension |
| `hidden_dim` | 2048 | Feed-forward network hidden dimension |
| `n_layers` | 12 | Number of transformer layers |
| `n_heads` | 12 | Number of attention heads |
| `n_kv_heads` | 12 | Number of key/value heads |
| `vocab_size` | 32000 | Vocabulary size |
| `seq_len` | 1024 | Maximum sequence length |
| `GS` | 64 | Quantization group size |

### Hardware Implementation Details

#### Memory Interface Design

The implementation uses a **flattened weight interface** where all weight tensors are passed as individual pointers, enabling: 

- **Parallel Weight Access**: Each weight array mapped to separate HBM banks (gmem0-gmem22)
- **Optimal Burst Patterns**: Sequential array access optimized for HBM burst reads/writes
- **Independent Bandwidth Control**: Per-interface depth and burst length specifications
- **Multi-Outstanding Transactions**: Up to 32 outstanding read requests for weight loading

#### HBM Bank Mapping

```
gmem0:  Token Embedding Table
gmem1:  Query Weights (wq)          | gmem2:  Query Scales
gmem3:  Key Weights (wk)            | gmem4:  Key Scales
gmem5:  Value Weights (wv)          | gmem6:  Value Scales
gmem7:  Output Attention (wo)       | gmem8:  Output Scales
gmem9:  FFN W1 Weights              | gmem10: FFN W1 Scales
gmem11: FFN W2 Weights              | gmem12: FFN W2 Scales
gmem13: FFN W3 Weights              | gmem14: FFN W3 Scales
gmem15: RMS Attention Weights       | gmem16: RMS FFN Weights
gmem17: RMS Final Weights           | gmem18: Classifier Weights
gmem19: Classifier Scales
gmem20: Key Cache                   | gmem21: Value Cache
gmem22: Output Logits
```

#### Core Operations

1. **RMSNorm**: Root Mean Square Layer Normalization with weight scaling
2. **Quantized Matrix Multiplication**: INT8 weights × FP32 activations with group-wise dequantization
3. **Multi-Head Attention**: With rotary positional embeddings (RoPE)
4. **SwiGLU Activation**: `silu(x) * linear(x)` for feed-forward networks
5. **Softmax**:  Temperature-scaled attention score normalization
6. **KV Caching**: Stores keys and values across sequence generation

## Repository Structure

```
HLSTranform/
├── code/
│   ├── src/                      # HLS kernel source code
│   │   ├── forward.cpp           # Main forward pass implementation
│   │   ├── forward.h             # Function declarations and helper functions
│   │   ├── config.h              # Model configuration parameters
│   │   └── typedefs.h            # Type definitions
│   ├── host/                     # Host-side application code
│   │   ├── llama2.cpp            # Main host application (generation & evaluation)
│   │   ├── forward.h             # Host-side function declarations
│   │   ├── config.h              # Host configuration (matches src/)
│   │   └── typedefs.h            # Host type definitions
│   └── testbench/                # HLS testbench and test data
│       ├── testbench.cpp         # C++ testbench for simulation
│       ├── tokenizer.bin         # Tokenizer model file
│       └── weights.bin           # Model weights checkpoint
├── compile. cfg                   # Vitis HLS compilation configuration
├── link.cfg                      # Vitis linker configuration
└── forward.ltx                   # LaTeX documentation/formulas
```

## Prerequisites

- **Xilinx Vitis HLS** 2021.2 or later
- **Xilinx Runtime (XRT)** for deployment
- **Alveo U50 FPGA** accelerator card
- **GCC/G++** with C++14 support
- Model checkpoint file (`weights.bin`) and tokenizer (`tokenizer.bin`)

## Building the Project

### 1. HLS Synthesis

Synthesize the C++ kernel to RTL:

```bash
v++ -c --mode hls --platform ${PLATFORM} --config hls_compile.cfg
```

This will: 
- Synthesize the `forward()` function targeting the Alveo U50
- Generate IP core with AXI interfaces
- Package the output as a `.xo` (Xilinx Object) file

### 2. Hardware Compilation

Compile the kernel to FPGA bitstream:

```bash
v++ -l  --platform ${PLATFORM}  --config link.cfg  -o forward.xclbin  forward.xo
```

### 3. Host Application

Compile the host application:

```bash
g++ host/llama2.cpp -o llama2_host \
    -I $XILINX_XRT/include \
    -I $VITIS_HLS/include \
    -L $XILINX_XRT/lib \
    -lxrt_coreutil -lxrt_core -lxrt_hwemu -luuid -pthread -std=c++17
```

## Usage

### Text Generation

```bash
./llama2 weights.bin -m generate -i "{USER_PROMPT}"
```

### Perplexity Evaluation

```bash
./llama2 weights. bin -m evaluate -e "{EVAL_FILE}"
```

## Acknowledgments

- Original HLSTransform implementation by He et al.
- Llama2.c model by Andrej Karpathy
- Xilinx/AMD for development tools and hardware platform

## Additional Resources

- [Xilinx Vitis HLS User Guide](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls)
- [Alveo U50 Data Sheet](https://www.xilinx.com/products/boards-and-kits/alveo/u50.html)
- [Llama 2 Model Card](https://github.com/facebookresearch/llama)
