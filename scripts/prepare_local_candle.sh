#!/usr/bin/env bash
set -euo pipefail

# Clone HuggingFace candle repository next to this project and run the
# small patch script to adjust candle-kernels build flags for local nvcc.

REPO=${CANDLE_REPO:-https://github.com/huggingface/candle}
DEST=${CANDLE_DEST:-../candle}

if [ -d "$DEST" ]; then
  echo "Destination already exists: $DEST"
  echo "If you want a fresh clone, remove it and re-run."
  exit 1
fi

echo "Cloning $REPO -> $DEST"
git clone "$REPO" "$DEST"

echo "Running patch script to adjust candle-kernels build flags"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
"$SCRIPT_DIR/patch_candle_kernels.sh" "$DEST"

echo
echo "Done. Next steps (on target device):"
echo "  1) Set CUDA env to your runtime (e.g. /usr/local/cuda-12.6)" 
echo "  2) In this project, add a patch to Cargo.toml to use local candle:" 
echo "     [patch.crates-io]" 
echo "     candle = { path = \"../candle\" }" 
echo "  3) Rebuild only the candle-kernels crate to regenerate PTX:" 
echo "     export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=\$CUDA_HOME/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" 
echo "     cargo clean -p candle-kernels || true" 
echo "     cargo build -p candle-kernels --release -v" 
echo "  4) Then rebuild your project: cargo build --release -v"
