#!/bin/bash
# Setup script for ONNX Runtime on Linux ARM64 / Jetson
# This installs system-wide ONNX Runtime and configures the Rust build

set -euo pipefail

echo "========================================="
echo "ONNX Runtime System Setup"
echo "========================================="

# Detect platform
if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    echo "Detected: Linux ARM64 (Jetson)"
elif [[ "$(uname -m)" == "x86_64" ]]; then
    echo "Detected: Linux x86_64"
else
    echo "Warning: Unknown architecture $(uname -m)"
fi

# Check if ONNX Runtime is already installed
if python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null; then
    echo "✓ ONNX Runtime Python package already installed"
    ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
    echo "  Location: $ORT_PYTHON_PATH"
else
    echo ""
    echo "Installing ONNX Runtime via pip..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip --quiet
    
    # Try to install onnxruntime
    if python3 -m pip install onnxruntime 2>/dev/null; then
        echo "✓ Standard onnxruntime installed"
        ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
        echo "  Location: $ORT_PYTHON_PATH"
    else
        echo ""
        echo "Standard install failed. Trying CUDA-enabled version..."
        
        # Check for CUDA
        if nvidia-smi &>/dev/null; then
            echo "CUDA detected. Installing onnxruntime-cuda-12 (or similar)..."
            
            # Try different CUDA versions - adjust based on your system
            if python3 -m pip install 'onnxruntime-gpu>=1.16' 2>/dev/null; then
                echo "✓ onnxruntime-gpu installed"
                ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
            elif python3 -m pip install 'onnxruntime-directml' 2>/dev/null; then
                echo "✓ onnxruntime-directml installed"
                ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
            else
                echo ""
                echo "Warning: Could not install CUDA-enabled ONNX Runtime via pip."
                echo "You may need to install from NVIDIA's wheel or build manually."
                
                # Fall back to CPU version as last resort
                if python3 -m pip install 'onnxruntime<1.17' 2>/dev/null; then
                    echo "✓ Fallback: onnxruntime CPU-only installed"
                    ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
                else
                    echo ""
                    echo "ERROR: Failed to install any ONNX Runtime version."
                    exit 1
                fi
            fi
        else
            echo "No CUDA detected. Installing CPU-only version..."
            python3 -m pip install onnxruntime --quiet
            ORT_PYTHON_PATH=$(python3 -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")
        fi
    fi
fi

# Find the ONNX Runtime library location
echo ""
echo "Finding ONNX Runtime library..."

# Method 1: Use Python to find it
if [[ -n "${ORT_PYTHON_PATH:-}" ]]; then
    # Try common locations based on Python package structure
    for lib_path in \
        "$ORT_PYTHON_PATH/../lib" \
        "$ORT_PYTHON_PATH/../../lib" \
        "/usr/lib" \
        "/usr/local/lib"; do
        
        if [[ -d "$lib_path" ]]; then
            # Look for ONNX Runtime libraries
            if ls "$lib_path"/libonnxruntime* 1>/dev/null 2>&1; then
                ORT_LIB_LOCATION="$lib_path"
                echo "✓ Found ONNX Runtime library in: $ORT_LIB_LOCATION"
                break
            fi
        fi
    done
    
    # If not found via Python, try common locations
    if [[ -z "${ORT_LIB_LOCATION:-}" ]]; then
        for lib_path in \
            "/usr/lib/aarch64-linux-gnu" \
            "/usr/local/lib" \
            "/opt/nvidia" \
            "$HOME/.local/lib"; do
            
            if ls "$lib_path"/libonnxruntime* 1>/dev/null 2>&1; then
                ORT_LIB_LOCATION="$lib_path"
                echo "✓ Found ONNX Runtime library in: $ORT_LIB_LOCATION"
                break
            fi
        done
    fi
fi

# If still not found, provide manual instructions
if [[ -z "${ORT_LIB_LOCATION:-}" ]]; then
    echo ""
    echo "⚠ Could not automatically find ONNX Runtime library."
    echo ""
    echo "Please install ONNX Runtime manually:"
    echo ""
    echo "Option 1: Install via pip (recommended for development)"
    echo "  pip3 install onnxruntime"
    echo ""
    echo "Option 2: Install from NVIDIA JetPack (for production on Jetson)"
    echo "  The ONNX Runtime libraries should be available in your JetPack installation."
    echo "  Common locations:"
    echo "    - /usr/lib/aarch64-linux-gnu/libonnxruntime.so"
    echo "    - /opt/nvidia/jetpack/lib/libonnxruntime.so"
    echo ""
    echo "Option 3: Build ONNX Runtime from source"
    echo "  See: https://github.com/microsoft/onnxruntime"
    echo ""
    echo "After installation, set:"
    echo "  export ORT_STRATEGY=system"
    echo "  export ORT_LIB_LOCATION=/path/to/libonnxruntime.so"
    echo ""
    
    # Try to find it ourselves one more time
    FOUND=$(find /usr -name "libonnxruntime*.so*" 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        ORT_LIB_LOCATION=$(dirname "$FOUND")
        echo ""
        echo "✓ Found via find command: $ORT_LIB_LOCATION"
    else
        echo "Please set ORT_STRATEGY=system and ORT_LIB_LOCATION before building."
        exit 1
    fi
fi

# Set environment variables for build
echo ""
echo "========================================="
echo "Build Configuration"
echo "========================================="
export ORT_STRATEGY=system
export ORT_LIB_LOCATION="$ORT_LIB_LOCATION"

echo "Exported environment variables:"
echo "  ORT_STRATEGY=$ORT_STRATEGY"
echo "  ORT_LIB_LOCATION=$ORT_LIB_LOCATION"

# Verify library exists
if ls "$ORT_LIB_LOCATION"/libonnxruntime* 1>/dev/null 2>&1; then
    echo ""
    echo "✓ Library verification successful:"
    ls -lh "$ORT_LIB_LOCATION"/libonnxruntime* | head -3
else
    echo ""
    echo "⚠ Warning: Could not verify library files in $ORT_LIB_LOCATION"
fi

# Build the project
echo ""
echo "========================================="
echo "Building qwen3_asr with ONNX Runtime"
echo "========================================="

cd /home/ray/Desktop/projects/qwen3-asr-rs/qwen3_asr_rs

cargo build --release --features onnx-runtime 2>&1 | tee /tmp/build_output.txt

if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo "✓ Build completed successfully!"
else
    echo ""
    echo "⚠ Build failed. Check /tmp/build_output.txt for details."
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "You can now run:"
echo "  cargo build --release --features onnx-runtime"
echo ""
echo "Or set these permanently in your ~/.bashrc or ~/.profile:"
echo "  export ORT_STRATEGY=system"
echo "  export ORT_LIB_LOCATION=$ORT_LIB_LOCATION"
