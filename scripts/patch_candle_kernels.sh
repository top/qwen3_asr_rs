#!/usr/bin/env bash
set -euo pipefail

# Usage: ./patch_candle_kernels.sh /path/to/candle
# This script edits candle-kernels/build.rs to add explicit nvcc/gencode args
# that help ensure PTX is generated for a target compute capability.

REPO_DIR=${1:-}
if [ -z "$REPO_DIR" ]; then
  echo "Usage: $0 /path/to/candle" >&2
  exit 2
fi

KERNELS_BUILD="$REPO_DIR/candle-kernels/build.rs"
if [ ! -f "$KERNELS_BUILD" ]; then
  echo "Cannot find $KERNELS_BUILD" >&2
  exit 3
fi

echo "Patching $KERNELS_BUILD to add explicit nvcc gencode args..."

PATCH_MARKER="// BEGIN_AUTOPATCH_NVCC_ARGS"

if grep -q "$PATCH_MARKER" "$KERNELS_BUILD"; then
  echo "Patch already applied; skipping." 
  exit 0
fi

# Insert gencode args before the call to build_ptx() by finding the line
# that contains ".build_ptx()" and adding args a few lines earlier.

awk -v marker="$PATCH_MARKER" '
{
  print $0
  if ($0 ~ /\.build_ptx\(\)/) {
    print "    // BEGIN_AUTOPATCH_NVCC_ARGS";
    print "    // Force nvcc to emit code suitable for Jetson (adjust compute as needed)";
    print "    .arg(\"-gencode=arch=compute_87,code=sm_87\")";
    print "    .arg(\"-gencode=arch=compute_87,code=compute_87\")";
    print "    // END_AUTOPATCH_NVCC_ARGS";
  }
}' "$KERNELS_BUILD" > "$KERNELS_BUILD.patched"

mv "$KERNELS_BUILD.patched" "$KERNELS_BUILD"
echo "Patched file written. Please review $KERNELS_BUILD if you want to adjust compute targets." 
