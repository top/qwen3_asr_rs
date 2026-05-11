# qwen3-asr-server

> [!TIP]
> **Switch to libtorch version**: [Click here to view the `main` branch](https://github.com/top/qwen3_asr_rs/tree/main). The `main` branch is a `libtorch`-based implementation, which has higher memory usage and slower inference speed.

OpenAI-compatible Audio Transcriptions API server based on `qwen3-asr-rs`. This version is powered by the **[Candle](https://github.com/huggingface/candle)** ML framework, providing a **lighter**, **faster**, and more **resource-efficient** inference experience. It is tuned for **Jetson Orin** (CUDA, compute capability 87 in the bundled scripts).

---

## Quick reference: what you need on disk

| Item | Purpose |
|------|---------|
| This repo (`qwen3_asr_rs`) | Server source |
| Sibling directory `candle/` with `candle-kernels/` | Required by `[patch.crates-io]` in `Cargo.toml` (local `candle-kernels` for CUDA / nvcc) |
| `Qwen3-ASR-0.6B/` (or any path you set in `MODEL_PATH`) | Hugging Face weights (`model.safetensors`, configs, vocab, …) |
| `tokenizer.json` inside that model directory | **Not** in the default HF snapshot; must be generated once (see below) |
| Optional: `tokenizer.sh` next to `qwen3_asr_rs` | If present, `build.sh` runs it when `tokenizer.json` is missing |

Recommended layout (example user `~/qwen3-asr`):

```text
~/qwen3-asr/
├── candle/                 # git clone huggingface/candle
│   └── candle-kernels/
├── qwen3_asr_rs/           # this repository
├── Qwen3-ASR-0.6B/         # snapshot of Qwen/Qwen3-ASR-0.6B (+ tokenizer.json)
└── tokenizer.sh            # optional; see "Tokenizer" and `build.sh`
```

**Important:** `Cargo.toml` patches `candle-kernels` with `path = "../candle/candle-kernels"`. That path is resolved from **`qwen3_asr_rs`**, so `candle` must sit **next to** `qwen3_asr_rs`, not inside it. If you clone only this repo into `~/qwen3_asr_rs` without a parent `candle`, compilation will fail until you fix the directory layout or change the patch path.

---

## 1. Clone Candle and (Jetson) patch kernels

From the **parent** of `qwen3_asr_rs` (e.g. `~/qwen3-asr`):

```bash
git clone https://github.com/huggingface/candle.git candle
```

On Jetson, adjust `candle-kernels/build.rs` for your GPU architecture. This repo ships helpers:

```bash
cd qwen3_asr_rs
./scripts/patch_candle_kernels.sh ../candle
```

The script inserts nvcc `-gencode` flags (default **sm_87** for Orin). Edit `scripts/patch_candle_kernels.sh` if your device uses a different compute capability.

Alternatively, use `./scripts/prepare_local_candle.sh` once (it clones to `../candle` and runs the patch). It refuses to overwrite an existing `../candle`.

Set CUDA for builds, for example:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

---

## 2. Download model weights

Example with the Hugging Face CLI:

```bash
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ~/qwen3-asr/Qwen3-ASR-0.6B
```

The snapshot includes `model.safetensors`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, etc. The Rust stack still expects a **`tokenizer.json`** file in the same folder (same convention as this repo’s CI). Without it you will see errors such as `tokenizer load failed: No such file or directory (os error 2)` after weights load.

### Generate `tokenizer.json`

**Option A — one-off Python** (from any machine with `transformers`):

```bash
python3 -c "
from transformers import AutoTokenizer
from pathlib import Path
model_dir = Path('~/qwen3-asr/Qwen3-ASR-0.6B').expanduser()
tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
tok.backend_tokenizer.save(str(model_dir / 'tokenizer.json'))
print('Wrote', model_dir / 'tokenizer.json')
"
```

If `local_files_only=True` fails (incomplete tree), omit it or use `from_pretrained('Qwen/Qwen3-ASR-0.6B', trust_remote_code=True)` and still `save()` into your local directory.

**Option B — `build.sh` automation:** `build.sh` checks for `tokenizer.json` under `MODEL_PATH` (default `../Qwen3-ASR-0.6B` relative to `qwen3_asr_rs`). If the file is missing, it runs **`../tokenizer.sh`** (sibling of `qwen3_asr_rs`). Put your venv + `pip install` + Python generation logic there, or symlink a script you keep elsewhere.

---

## 3. Build and run

### Full install script (Jetson / production-style)

From inside `qwen3_asr_rs`:

```bash
./build.sh
```

This script:

1. Ensures `tokenizer.json` exists (via `../tokenizer.sh` if needed; see above).
2. Installs system packages (cmake, FFmpeg dev libs, …).
3. `git pull` (best-effort).
4. `cargo build --release`.
5. Copies `qwen3-asr-server.service` into `~/.config/systemd/user/`, reloads systemd, enables/restarts the user service.

Edit **`qwen3-asr-server.service`** so `WorkingDirectory`, `MODEL_PATH`, and `ExecStart` match your real paths before relying on systemd.

### User systemd service notes

- Install unit: `cp qwen3-asr-server.service ~/.config/systemd/user/`
- `systemctl --user daemon-reload`
- For services to survive logout (typical on a headless box): `sudo loginctl enable-linger $USER`
- `systemctl --user enable --now qwen3-asr-server`
- Logs: `journalctl --user-unit=qwen3-asr-server -f`

### Manual / dev (`start.sh`)

`start.sh` sets Jetson-oriented defaults (port **11435**, `MODEL_PATH` default `../Qwen3-ASR-0.6B/`, CUDA 12.6 paths) and runs `cargo run --release`. Run it from the `qwen3_asr_rs` directory.

---

## 4. Configuration

Environment variables (see `src/config.rs`; defaults below apply when unset):

| Variable | Default in code | Typical deployment |
|----------|-----------------|----------------------|
| `PORT` | `8080` | `11435` in `start.sh` / example service |
| `CONCURRENCY_LIMIT` | `2` | Same |
| `MODEL_PATH` | `models` | Absolute path to `Qwen3-ASR-0.6B` (must contain `tokenizer.json`) |
| `MODEL_NAME` | basename of `MODEL_PATH` | Set explicitly if API `model=` must match a fixed id |
| `CUDA_DEVICE` | *(not read in `config.rs`; used by upstream / env)* | `true` in scripts |

`JETSON_TARGET`, `CUDA_HOME`, `CUDA_COMPUTE_CAP`, and `RUST_LOG` are set in shell/service files for tooling and logging; adjust to match your CUDA install.

---

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

---

## Project structure

```
qwen3_asr_rs/
├── src/
│   ├── api/              # API handlers and types
│   ├── audio/            # Audio processing
│   ├── inference/        # Inference engine
│   ├── concurrency/      # Rate limiting
│   ├── config.rs         # Configuration
│   └── main.rs           # Server entry point
├── scripts/
│   ├── prepare_local_candle.sh   # clone ../candle + patch
│   └── patch_candle_kernels.sh   # nvcc gencode patch for candle-kernels
├── tests/
├── build.sh              # apt deps, tokenizer check, release build, systemd
├── start.sh              # dev-oriented env + cargo run
├── qwen3-asr-server.service
├── Dockerfile
├── Cargo.toml            # candle-kernels path patch
└── README.md
```

---

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI-style Audio Transcriptions
- **WAV support**: Optimized for WAV format processing
- **SSE streaming**: Real-time transcription with Server-Sent Events
- **CUDA acceleration**: NVIDIA GPU inference via Candle + `qwen3-asr`
- **Concurrency control**: Reduces risk of GPU OOM on small devices
- **Docker support**: Optional container workflow (see `Dockerfile`)

---

## API (short)

### `GET /v1/models`

Lists the configured model (`MODEL_NAME` or basename of `MODEL_PATH`).

### `POST /v1/audio/transcriptions`

Multipart form: `file` (WAV), `model` (must match configured name), optional `language`, `prompt`, `response_format`, `temperature`, `stream`.

Example:

```bash
curl -X POST "http://127.0.0.1:11435/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B" \
  -F "language=en"
```

---

## Docker (Jetson)

```bash
docker build -t qwen3-asr-server .
docker run -d -p 8080:8080 -v /path/to/models:/app/models --gpus all qwen3-asr-server
```

Mount a directory that includes **`tokenizer.json`** alongside the weights.

---

## Troubleshooting

| Symptom | Likely cause |
|--------|----------------|
| Cargo cannot find `candle-kernels` path dependency | `../candle/candle-kernels` missing; clone Candle next to `qwen3_asr_rs` or change `[patch.crates-io]` |
| nvcc / PTX / GPU arch errors | Wrong `-gencode` in patched `build.rs`; CUDA version vs `cudarc` features in `Cargo.toml` |
| Weights load then tokenizer `os error 2` | Missing `tokenizer.json` in `MODEL_PATH` |
| systemd starts wrong binary or model | Unit file still points at old `WorkingDirectory` / `MODEL_PATH` / `ExecStart` |

---

## Known upstream notes

- Candle / CUDA versions vary by board and JetPack; the local `candle-kernels` patch exists to align nvcc with your device.
- CI in this repo generates `tokenizer.json` for tests; production snapshots from Hugging Face alone are not sufficient without that step.

---

## Development status

- [x] US-001: Project foundation and Axum server
- [ ] US-002: Audio processing module (WAV only)
- [ ] US-003: Candle-based inference (CUDA)
- [ ] US-004: OpenAI parameters and JSON response
- [ ] US-005: SSE streaming
- [ ] US-006: VRAM-safe concurrency
- [x] US-007: Containerization for Jetson

---

## License

MIT
