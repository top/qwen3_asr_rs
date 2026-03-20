# Build stage
FROM nvcr.io/nvidia/l4t-rust:latest AS builder

WORKDIR /app

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Build the project
RUN cargo build --release

# Runtime stage
FROM nvcr.io/nvidia/l4t-base:r36.2.0

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built binary
COPY --from=builder /app/target/release/qwen3-asr-server /app/qwen3-asr-server

# Create model directory
RUN mkdir -p /app/models

# Set environment variables
ENV PORT=8080
ENV CONCURRENCY_LIMIT=2
ENV MODEL_PATH=/app/models
ENV CUDA_DEVICE=true

# Expose port
EXPOSE 8080

# Run the server
CMD ["/app/qwen3-asr-server"]
