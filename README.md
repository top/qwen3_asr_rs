# Qwen3 ASR -- Rust CLI tools

Pure Rust implementation of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) automatic speech recognition. The project builds a cross-platform CLI tool suitable for agentic skills for AI agents and bots.

- **asr** generates text from an input audio file (supports most codex and file formats)

Supports two backends: **libtorch** (via the `tch` crate, cross-platform with optional CUDA) and **MLX** (Apple Silicon native via Metal GPU). Loads model weights directly from safetensors files and re-implements the complete neural network forward pass in Rust.

## Quick Start

### 1. Download the release

Download the latest release for your platform from [GitHub Releases](https://github.com/second-state/qwen3_asr_rs/releases/latest) and extract. Each zip contains the `asr` binary and all required runtime dependencies (libtorch for Linux, mlx.metallib for macOS).

**macOS (Apple Silicon)**

```bash
curl -LO https://github.com/second-state/qwen3_asr_rs/releases/latest/download/asr-macos-aarch64.zip
unzip asr-macos-aarch64.zip
# Contains: asr-macos-aarch64/asr, asr-macos-aarch64/mlx.metallib
```

**Linux x86_64 (CPU)**

```bash
curl -LO https://github.com/second-state/qwen3_asr_rs/releases/latest/download/asr-linux-x86_64.zip
unzip asr-linux-x86_64.zip
# Contains: asr-linux-x86_64/asr, asr-linux-x86_64/libtorch/
```

**Linux x86_64 (CUDA)**

```bash
curl -LO https://github.com/second-state/qwen3_asr_rs/releases/latest/download/asr-linux-x86_64-cuda.zip
unzip asr-linux-x86_64-cuda.zip
# Contains: asr-linux-x86_64-cuda/asr, asr-linux-x86_64-cuda/libtorch/
```

**Linux ARM64**

```bash
curl -LO https://github.com/second-state/qwen3_asr_rs/releases/latest/download/asr-linux-aarch64.zip
unzip asr-linux-aarch64.zip
# Contains: asr-linux-aarch64/asr, asr-linux-aarch64/libtorch/
```

### 2. Download model weights

```bash
pip install huggingface_hub transformers

huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B

python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen3-ASR-0.6B', trust_remote_code=True)
tok.backend_tokenizer.save('Qwen3-ASR-0.6B/tokenizer.json')
"
```

### 3. Transcribe

```bash
# macOS
./asr-macos-aarch64/asr Qwen3-ASR-0.6B input.wav

# Linux
./asr-linux-x86_64/asr Qwen3-ASR-0.6B input.wav
```

Output:

```
Language: English
Text: Thank you for your contribution to the most recent issue of Computer.
```

## Architecture

The implementation ports the Qwen3-ASR encoder-decoder architecture from PyTorch/Transformers to Rust with libtorch (via the `tch` crate):

- **Audio Encoder** (Whisper-style): 3x Conv2d downsampling → sinusoidal positional embeddings → 18 transformer encoder layers → output projection (896 → 1024)
- **Text Decoder** (Qwen3): 28 transformer decoder layers with Grouped Query Attention (16 Q heads / 8 KV heads), QK-normalization, MRoPE (Multimodal Rotary Position Embeddings), and SwiGLU MLP
- **Audio preprocessing**: FFmpeg decodes any audio format → resampled to mono 16kHz f32 → 128-bin log-mel spectrogram (Whisper-style)

## Supported Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Qwen3-ASR-0.6B | 0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ASR-1.7B | 1.7B | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

## Usage

```bash
# Basic transcription (auto-detect language)
asr ./Qwen3-ASR-0.6B input.wav

# Force language
asr ./Qwen3-ASR-0.6B input.wav chinese
asr ./Qwen3-ASR-0.6B input.wav english

# Enable debug logging
RUST_LOG=debug asr ./Qwen3-ASR-0.6B input.wav
```

### Output Format

```
Language: Chinese
Text: 你好世界
```

## Supported Languages

Qwen3-ASR supports 30 languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Build from Source

### Backend

Choose one backend:

| Backend | Feature flag | Platforms | GPU |
|---------|-------------|-----------|-----|
| libtorch | `tch-backend` (default) | Linux, macOS, Windows | CUDA |
| MLX | `mlx` | macOS Apple Silicon | Metal |

### Prerequisites

**libtorch** (for `tch-backend`): Download and extract for your platform:

```bash
# Linux x86_64 (CPU)
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cpu.zip

# Linux ARM64 (CPU)
curl -LO https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz
tar xzf libtorch-cxx11-abi-aarch64-2.7.1.tar.gz

# Linux x86_64 (CUDA 12.8)
curl -LO https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip
```

**FFmpeg** development libraries:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswresample-dev pkg-config
```

### libtorch backend (default)

```bash
# Set environment
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Build FFmpeg from source and link statically (most self-contained)
cargo build --release --features build-ffmpeg
```

### MLX backend (macOS Apple Silicon)

```bash
# Initialize mlx-c submodule
git submodule update --init --recursive

# Build with MLX (no libtorch needed)
cargo build --release --no-default-features --features mlx,build-ffmpeg
```

## Project Structure

```
src/
├── main.rs            # CLI binary entry point
├── lib.rs             # Library module declarations
├── tensor.rs          # Unified Tensor abstraction (tch/MLX backend)
├── config.rs          # Model configuration (from config.json)
├── error.rs           # Error types
├── audio.rs           # FFmpeg-based audio loading and format conversion
├── mel.rs             # Whisper-style mel spectrogram feature extraction
├── weights.rs         # Safetensors weight loading (bf16 → f32 conversion)
├── layers.rs          # Neural network building blocks (LayerNorm, RMSNorm,
│                      #   attention, MLP, MRoPE, etc.)
├── audio_encoder.rs   # Whisper-style audio encoder (Conv2d + Transformer)
├── text_decoder.rs    # Qwen3 text decoder with KV cache
├── tokenizer.rs       # HuggingFace tokenizer wrapper
├── inference.rs       # End-to-end ASR inference pipeline
└── backend/
    └── mlx/           # Apple MLX backend (Metal GPU)
        ├── ffi.rs     # Raw C FFI bindings to mlx-c
        ├── array.rs   # Safe RAII MlxArray wrapper
        ├── ops.rs     # Safe operation wrappers
        ├── io.rs      # Safetensors loading via mlx-c
        ├── signal.rs  # STFT, mel spectrogram signal processing
        └── stream.rs  # Device/stream management
```

## License

Apache-2.0
