#!/usr/bin/env python3
"""
Export Qwen3 ASR model to ONNX format for Rust ONNX Runtime backend testing.

This script exports the audio encoder and text decoder components separately,
as they have different input/output requirements.

Usage:
    python export_to_onnx.py --model ./Qwen3-ASR-0.6B --output-dir ./onnx_models

Options:
    --model: Path to downloaded Qwen3 ASR model directory
    --output-dir: Output directory for ONNX files (default: ./onnx_models)
    --audio-only: Export only the audio encoder component
    --decoder-only: Export only the text decoder component
"""

import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2seq, AutoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_audio_encoder(model, processor, output_path: str, device: str = "cpu"):
    """Export the audio encoder component to ONNX."""
    logger.info("Exporting audio encoder...")

    # Create dummy input (audio features are typically 80 mel bins)
    batch_size = 1
    seq_len = 3000  # Approximate max sequence length for audio features

    dummy_input = torch.randn(batch_size, seq_len, dtype=torch.float32).to(device)

    # Get model components
    audio_encoder = model.audio_encoder

    # Export to ONNX
    torch.onnx.export(
        audio_encoder,
        dummy_input,
        output_path,
        input_names=["input_features"],
        output_names=["audio_tokens"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "audio_tokens": {0: "batch_size", 1: "token_count"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    logger.info(f"Audio encoder exported to {output_path}")


def export_text_decoder(model, processor, output_path: str, device: str = "cpu"):
    """Export the text decoder component to ONNX."""
    logger.info("Exporting text decoder...")

    # Create dummy inputs
    batch_size = 1
    seq_len = 50

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long).to(
        device
    )
    attention_mask = torch.ones_like(input_ids).to(device)

    # Get model components
    text_decoder = model.text_decoder

    # Export to ONNX
    torch.onnx.export(
        text_decoder,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    logger.info(f"Text decoder exported to {output_path}")


def export_full_model(model, processor, output_path: str, device: str = "cpu"):
    """Export the full Qwen3 ASR model to ONNX."""
    logger.info("Exporting full Qwen3 ASR model...")

    # Process a sample audio file or create dummy inputs
    batch_size = 1

    # Create dummy audio input (raw waveform)
    audio_input = torch.randn(batch_size, 16000, dtype=torch.float32).to(device)
    input_features = model.feature_extractor(audio_input).last_hidden_state.to(device)

    # Generate some dummy text IDs for decoder warmup
    input_ids = processor.batch_decode([["hello world"]])[0]
    input_ids = (
        torch.tensor(processor.text_to_tokens(input_ids).input_ids, dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )
    attention_mask = torch.ones_like(input_ids).to(device)

    # Export the full model
    try:
        torch.onnx.export(
            model,
            (audio_input,),
            output_path,
            input_names=["input_features"],
            output_names=["output_tokens", "logits"],
            dynamic_axes={
                "input_features": {0: "batch_size"},
                "output_tokens": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        logger.info(f"Full model exported to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to export full model: {e}")
        logger.info("Try exporting components separately instead")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3 ASR model to ONNX format for Rust ONNX Runtime backend testing."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to downloaded Qwen3 ASR model directory (e.g., ./Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_models",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Export only the audio encoder component",
    )
    parser.add_argument(
        "--decoder-only",
        action="store_true",
        help="Export only the text decoder component",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export (default: cpu)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.isdir(args.model):
        logger.error(f"Model directory does not exist: {args.model}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    logger.info(f"Loading Qwen3 ASR model from {args.model}...")
    try:
        model = AutoModelForSpeechSeq2seq.from_pretrained(
            args.model,
            torch_dtype=torch.float32,  # Use float32 for ONNX compatibility
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = AutoProcessor.from_pretrained(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    model.to(args.device)
    model.eval()

    # Export based on flags
    if args.audio_only:
        export_audio_encoder(
            model, processor, str(output_dir / "audio_encoder.onnx"), device=args.device
        )
    elif args.decoder_only:
        export_text_decoder(
            model, processor, str(output_dir / "text_decoder.onnx"), device=args.device
        )
    else:
        # Export both components separately (recommended for Rust backend)
        export_audio_encoder(
            model, processor, str(output_dir / "audio_encoder.onnx"), device=args.device
        )

        export_text_decoder(
            model, processor, str(output_dir / "text_decoder.onnx"), device=args.device
        )

    logger.info("Export completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
