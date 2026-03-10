# Qwen3 ASR -- Rust CLI & API Server

Pure Rust, highly optimized implementation of [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) automatic speech recognition. This project provides a cross-platform CLI tool and an OpenAI-compatible API server, specifically optimized for extremely low-memory edge devices like the NVIDIA Jetson Orin Nano.

## 🚀 Key Optimizations for Edge Devices (e.g., NVIDIA Jetson)

To run a 1B+ parameter Audio-Language Model on a 4GB/8GB unified memory chip without Out-of-Memory (OOM) errors:
- **`no_grad()` Inference**: Fully disabled PyTorch backpropagation graph building to save gigabytes of VRAM.
- **Zero-Copy Memory Mapping (`memmap2`)**: Safetensors weights are memory-mapped directly from the disk to the GPU, eliminating the massive CPU RAM spike (2GB+) during model loading.
- **Native BFloat16/FP16 Loading**: Weights bypass FP32 decompression, preventing VRAM doubling and locking the model footprint to its true minimal size (~1.2GB).
- **Dynamic Type Casting**: Audio features and attention masks automatically match the natively loaded `BFloat16` weights to prevent PyTorch type mismatches.

## Quick Start on NVIDIA Jetson / Linux

This fork bypasses fragile `pip` installations by directly downloading and linking the pre-compiled NVIDIA PyTorch wheel.

### 1. Download Model and Tokenizer
You need to pull the HuggingFace weights and generate the `tokenizer.json` file which the Rust `tokenizers` crate requires:

```bash
# In a Python environment (like venv) with transformers installed:
pip install huggingface_hub transformers

# Download the model
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir Qwen3-ASR-0.6B

# Extract the Rust-compatible tokenizer.json
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./Qwen3-ASR-0.6B', trust_remote_code=True)
tok.backend_tokenizer.save('./Qwen3-ASR-0.6B/tokenizer.json')
"
```

### 2. Install Dependencies
```bash
sudo apt-get update -qq
sudo apt-get install -y cmake pkg-config g++ nasm unzip curl tar
```

If you need to decode non-WAV formats (MP3/FLAC/AAC/OGG/etc.), install system
FFmpeg development packages too:

```bash
sudo apt-get install -y libavutil-dev libavformat-dev libavcodec-dev libavdevice-dev \
   libavfilter-dev libswscale-dev libswresample-dev
```

### 3. Setup PyTorch (Jetpack 6.0 / ARM64)
```bash
# Download and extract the PyTorch wheel for Jetson
curl -L -o torch-2.3.0.whl https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl
unzip -qo torch-2.3.0.whl -d .torch

# Export environment variables for the compiler and runtime
export LIBTORCH="$(pwd)/.torch/torch"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
```

### 4. Build from Source
```bash
git submodule update --init --recursive

# Option A — WAV-only (recommended if you only use WAV files):
# No FFmpeg required; build normally.
cargo build --release

# Option B — Full format support (MP3/FLAC/AAC/OGG/etc):
# Enable FFmpeg support (uses system FFmpeg libraries if available):
cargo build --release --features ffmpeg

# Advanced: build static FFmpeg (heavy, rarely needed):
# cargo build --release --features build-ffmpeg
```

## Usage: OpenAI-Compatible API Server

Start the lightweight Actix-Web HTTP server for handling audio transcriptions via API.

Audio format notes:
- WAV files: supported natively (no FFmpeg required).
- Other formats (MP3/FLAC/AAC/OGG/etc): require FFmpeg — either the system FFmpeg
   dev packages or building with `--features build-ffmpeg`.

Example (start server):

```bash
./target/release/asr serve ./Qwen3-ASR-0.6B -p 11435 --backup-dir ./backup/audio/ --db-path ./backup/asr.db
```

Streaming: the `/v1/audio/transcriptions` endpoint accepts an optional multipart
field `stream` (values `true`, `1`, `yes`). When set, the server responds with
Server-Sent Events (SSE) and sends partial transcription updates as they are
generated, then a final transcription.

This starts a `/v1/audio/transcriptions` endpoint capable of processing standard
Whisper-style API requests. It records API metadata to SQLite (`backup/asr.db`) and
saves incoming audio locally.

## Systemd Service Deployment (Optional)

To run the API server continuously in the background and start it on boot, you can configure it as a systemd user service.

1. **Install and start the service**:
   ```bash
   mkdir -p ~/.config/systemd/user/
   cp qwen3-asr-server.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now qwen3-asr-server
   ```

2. **Check status and logs**:
   ```bash
   systemctl --user status qwen3-asr-server
   journalctl --user -u qwen3-asr-server -n 80 -f
   ```

## Usage: CLI Tool

```bash
# Basic transcription (auto-detect language)
./target/release/asr transcribe ./Qwen3-ASR-0.6B input.wav

# Force language
./target/release/asr transcribe ./Qwen3-ASR-0.6B input.wav chinese

# Enable debug logging
RUST_LOG=debug ./target/release/asr transcribe ./Qwen3-ASR-0.6B input.wav
```

## Supported Models

| Model | Parameters | HuggingFace |
|-------|-----------|-------------|
| Qwen3-ASR-0.6B | 0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ASR-1.7B | 1.7B | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

## Supported Languages

Qwen3-ASR supports 30 languages: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Architecture

The implementation completely maps the Qwen3-ASR architecture using Rust libtorch bindings:

- **Audio Encoder** (Whisper-style): 3x Conv2d downsampling → sinusoidal positional embeddings → 18 transformer encoder layers
- **Text Decoder** (Qwen3): 28 transformer decoder layers with Grouped Query Attention, QK-normalization, MRoPE, and SwiGLU MLP
- **Audio preprocessing**: FFmpeg decodes any format → resamples to mono 16kHz f32 → 128-bin log-mel spectrogram

## License

Apache-2.0
