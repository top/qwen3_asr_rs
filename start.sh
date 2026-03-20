#!/bin/bash

# qwen3-asr-server startup script for Jetson Orin Nano

# Set environment variables
export PORT=${PORT:-11433}
export CONCURRENCY_LIMIT=${CONCURRENCY_LIMIT:-2}
export MODEL_PATH=${MODEL_PATH:-../qwen3-asr-rs/Qwen3-ASR-0.6B/}
export CUDA_DEVICE=${CUDA_DEVICE:-true}
export JETSON_TARGET=true
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_COMPUTE_CAP=87
export RUST_LOG=${RUST_LOG:-info}

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory not found: $MODEL_PATH"
    echo "Please download the qwen3-asr model first:"
    echo "  huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir $MODEL_PATH"
    exit 1
fi

# Check if binary exists, if not build it
if [ ! -f "target/release/qwen3-asr-server" ]; then
    echo "Building qwen3-asr-server..."
    cargo build --release
fi

# Run the server
cargo run --release -j "$(nproc)"
