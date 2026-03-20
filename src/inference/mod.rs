use anyhow::Result;
use qwen3_asr::{AsrInference, TranscribeOptions};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;

pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
}

pub struct InferenceEngine {
    engine: Arc<AsrInference>,
}

impl InferenceEngine {
    pub fn new(config: &Config) -> Result<Self> {
        info!("Loading qwen3-asr model from: {}", config.model_path);

        let device = qwen3_asr::best_device();
        info!("Using device: {:?}", device);

        let engine = AsrInference::load(std::path::Path::new(&config.model_path), device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        info!("Model loaded successfully");
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<TranscriptionResult> {
        // Handle empty audio data
        if audio_data.is_empty() {
            return Err(anyhow::anyhow!("No audio samples to transcribe"));
        }

        // Handle very short audio (less than 0.1 seconds at 16kHz)
        if audio_data.len() < 1600 {
            tracing::warn!(
                "Very short audio: {} samples ({} seconds)",
                audio_data.len(),
                audio_data.len() as f64 / 16000.0
            );
        }

        info!(
            "Starting transcription of {} samples ({:.2}s)",
            audio_data.len(),
            audio_data.len() as f64 / 16000.0
        );

        let options = TranscribeOptions::default();
        let result = self
            .engine
            .transcribe_samples(audio_data, options)
            .map_err(|e| anyhow::anyhow!("Transcription failed: {}", e))?;

        info!("Transcription completed: {} chars", result.text.len());

        Ok(TranscriptionResult {
            text: result.text,
            language: result.language,
        })
    }

    pub fn init_streaming(&self, options: qwen3_asr::StreamingOptions) -> qwen3_asr::StreamingState {
        self.engine.init_streaming(options)
    }

    pub fn feed_audio(
        &self,
        state: &mut qwen3_asr::StreamingState,
        samples: &[f32],
    ) -> Result<Option<TranscriptionResult>> {
        let result = self.engine.feed_audio(state, samples)
            .map_err(|e| anyhow::anyhow!("Streaming feed failed: {}", e))?;
        
        Ok(result.map(|r| TranscriptionResult {
            text: r.text,
            language: r.language,
        }))
    }

    pub fn finish_streaming(
        &self,
        state: &mut qwen3_asr::StreamingState,
    ) -> Result<TranscriptionResult> {
        let result = self.engine.finish_streaming(state)
            .map_err(|e| anyhow::anyhow!("Streaming finish failed: {}", e))?;
        
        Ok(TranscriptionResult {
            text: result.text,
            language: result.language,
        })
    }
}
