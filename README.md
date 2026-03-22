# qwen3-asr-server

> [!TIP]
> **Switch to libtorch version**: [Click here to view the `main` branch](https://github.com/top/qwen3_asr_rs/tree/main). The `main` branch is a `libtorch`-based implementation, which has higher memory usage and slower inference speed.

OpenAI-compatible Audio Transcriptions API server based on `qwen3-asr-rs`. This version is powered by the **[Candle](https://github.com/huggingface/candle)** ML framework, providing a **lighter**, **faster**, and more **resource-efficient** inference experience. Optimized specifically for Jetson Orin Nano with CUDA acceleration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (Axum)                       │
├─────────────────────────────────────────────────────────────┤
│  POST /v1/audio/transcriptions                              │
│  ├── Normal Mode: JSON response                             │
│  └── Streaming Mode: SSE (Server-Sent Events)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Request Processing                         │
├─────────────────────────────────────────────────────────────┤
│  - Multipart form data parsing                              │
│  - WAV format validation                                    │
│  - Parameter extraction (OpenAI compatible)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Audio Processing Module                     │
├─────────────────────────────────────────────────────────────┤
│  AudioProcessor:                                            │
│  - Extract raw PCM data                                     │
│  - Get sample rate, channels                                │
│  - Validate WAV format                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                Concurrency Control Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Semaphore-based rate limiter                               │
│  - Limit concurrent inference tasks                         │
│  - Queue incoming requests                                  │
│  - Configurable via environment variable                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inference Engine (CUDA)                      │
├─────────────────────────────────────────────────────────────┤
│  InferenceEngine:                                           │
│  - Load qwen3-asr 0.6B model                                │
│  - Use candle framework with CUDA backend                   │
│  - Execute transcription (batch/streaming)                  │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
qwen3-asr-server/
├── src/
│   ├── api/              # API handlers and types
│   ├── audio/            # Audio processing
│   ├── inference/        # Inference engine
│   ├── concurrency/      # Rate limiting
│   ├── config.rs         # Configuration
│   └── main.rs           # Server entry point
├── tests/                # Integration tests
├── Dockerfile            # Jetson deployment
├── Cargo.toml
└── .env.example
```

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI Audio Transcriptions
- **WAV support**: Optimized for WAV format processing
- **SSE streaming**: Real-time transcription with Server-Sent Events
- **CUDA acceleration**: Leverages NVIDIA GPU for fast inference
- **Concurrency control**: Prevents GPU OOM on resource-constrained devices
- **Docker support**: Easy deployment on Jetson Orin Nano

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `CONCURRENCY_LIMIT` | 2 | Max concurrent requests |
| `MODEL_PATH` | models | Path to model directory |
| `MODEL_NAME` | *(auto)* | Explicit model name for API validation (defaults to `MODEL_PATH` basename) |
| `CUDA_DEVICE` | true | Use CUDA device |

## API Endpoints

### GET /v1/models

Lists the currently available model (determined by `MODEL_NAME` or `MODEL_PATH`).

**Response (JSON):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-ASR-0.6B",
      "object": "model",
      "created": 1711100000,
      "owned_by": "openai"
    }
  ]
}
```

### POST /v1/audio/transcriptions

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | WAV audio file |
| `model` | string | Yes | Model name (MUST match `MODEL_NAME` or `MODEL_PATH` basename) |
| `language` | string | No | Language code (e.g., "en", "zh") |
| `prompt` | string | No | Contextual prompt |
| `response_format` | string | No | "json" or "text" |
| `temperature` | number | No | Sampling temperature |
| `stream` | boolean | No | Enable SSE streaming |

**Response (JSON):**

```json
{
  "text": "Transcribed text",
  "language": "en",
  "duration": 10.5
}
```

**Response (SSE):**

```
data: {"text": "partial text", "language": "en", "duration": 10.5}
```

## Usage

### Local Development

```bash
# Install dependencies
cargo build

# Run server
cargo run

# Test with curl (assuming MODEL_NAME=Qwen3-ASR-0.6B)
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B" \
  -F "language=en"
```

### Docker (Jetson Orin Nano)

```bash
# Build image
docker build -t qwen3-asr-server .

# Run container
docker run -d \
  -p 8080:8080 \
  -v /path/to/models:/app/models \
  --gpus all \
  qwen3-asr-server
```

## Known Issues

1. **CUDA compilation error**: The `candle-kernels` crate has compatibility issues with CUDA 12.4 on some systems. This is a known issue with the upstream library and does not affect Jetson Orin Nano deployment.

2. **Model download**: Models must be downloaded manually to the `MODEL_PATH` directory before running.

## Development Status

- [x] US-001: Project Foundation & Basic Axum Server
- [ ] US-002: Audio Processing Module (WAV Only)
- [ ] US-003: Candle-based Inference Engine (CUDA)
- [ ] US-004: OpenAI Parameter Support & JSON Response
- [ ] US-005: SSE Single-Direction Streaming
- [ ] US-006: VRAM-Safe Concurrency (Semaphore)
- [x] US-007: Containerization for Jetson

## License

MIT
