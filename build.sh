#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

sudo apt-get update -qq
sudo apt-get install -y -qq cmake pkg-config g++ nasm unzip \
    libavutil-dev libavformat-dev libavcodec-dev libavdevice-dev \
    libavfilter-dev libswscale-dev libswresample-dev curl tar

git fetch --quiet --all || true
git pull --ff-only || true
cargo build --release -j $(nproc)

cp -f qwen3-asr-server.service ~/.config/systemd/user/ 2>/dev/null
systemctl --user daemon-reload
systemctl --user enable qwen3-asr-server || true
systemctl --user restart qwen3-asr-server || true

echo "Build finished."
echo "  Check status:   systemctl --user status qwen3-asr-server"
echo "  View logs:      journalctl --user-unit qwen3-asr-server -f"
echo "  GPU usage:      nvidia-smi -l 2"
