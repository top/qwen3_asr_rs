use anyhow::Result;
use qwen3_asr::{AsrInference, TranscribeOptions};
use tracing::info;

use crate::config::Config;

pub struct TranscriptionResult {
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct InferenceOptions {
    pub language: Option<String>,
    pub context_text: Option<String>,
}

pub struct InferenceEngine {
    engine: AsrInference,
}

impl InferenceEngine {
    pub fn new(config: &Config) -> Result<Self> {
        info!("Loading qwen3-asr model from: {}", config.model_path);

        let device = qwen3_asr::best_device();
        info!("Using device: {:?}", device);

        let engine = AsrInference::load(std::path::Path::new(&config.model_path), device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        info!("Model loaded successfully");
        Ok(Self { engine })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<TranscriptionResult> {
        self.transcribe_with_options(audio_data, &InferenceOptions::default())
    }

    pub fn transcribe_with_options(
        &self,
        audio_data: &[f32],
        options: &InferenceOptions,
    ) -> Result<TranscriptionResult> {
        if audio_data.is_empty() {
            return Err(anyhow::anyhow!("No audio samples to transcribe"));
        }

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

        // If context is provided, use streaming path with initial_text so we can
        // condition decoding without forking qwen3-asr.
        if options
            .context_text
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
        {
            return self.transcribe_with_streaming_context(audio_data, options);
        }

        let sample_rate = 16000;
        let segments = crate::audio::segment_audio(audio_data, sample_rate, 20.0);
        info!(
            "Audio segmented into {} chunks for VRAM optimization",
            segments.len()
        );

        let mut final_text = String::new();

        for (i, segment) in segments.iter().enumerate() {
            info!("Processing segment {}/{}", i + 1, segments.len());
            let mut decode_opts = TranscribeOptions::default();
            decode_opts.language = options.language.clone();
            let result = self
                .engine
                .transcribe_samples(segment, decode_opts)
                .map_err(|e| anyhow::anyhow!("Transcription failed on segment {}: {}", i + 1, e))?;

            if !final_text.is_empty() && !result.text.is_empty() {
                final_text.push(' ');
            }
            final_text.push_str(&result.text);
        }

        info!("Transcription completed: {} chars", final_text.len());

        Ok(TranscriptionResult { text: final_text })
    }

    fn transcribe_with_streaming_context(
        &self,
        audio_data: &[f32],
        options: &InferenceOptions,
    ) -> Result<TranscriptionResult> {
        let mut streaming_opts = qwen3_asr::StreamingOptions::default();
        if let Some(language) = options.language.as_deref() {
            streaming_opts = streaming_opts.with_language(language.to_string());
        }
        if let Some(context_text) = options.context_text.as_deref() {
            let trimmed = context_text.trim();
            if !trimmed.is_empty() {
                streaming_opts = streaming_opts.with_initial_text(trimmed.to_string());
            }
        }

        let mut state = self.engine.init_streaming(streaming_opts);

        const CHUNK_SIZE: usize = 19200; // 1.2 s at 16 kHz
        for chunk in audio_data.chunks(CHUNK_SIZE) {
            self.engine
                .feed_audio(&mut state, chunk)
                .map_err(|e| anyhow::anyhow!("Streaming feed failed: {}", e))?;
        }

        let result = self
            .engine
            .finish_streaming(&mut state)
            .map_err(|e| anyhow::anyhow!("Streaming finish failed: {}", e))?;

        Ok(TranscriptionResult { text: result.text })
    }

    pub fn init_streaming(
        &self,
        options: qwen3_asr::StreamingOptions,
    ) -> qwen3_asr::StreamingState {
        self.engine.init_streaming(options)
    }

    pub fn feed_audio(
        &self,
        state: &mut qwen3_asr::StreamingState,
        samples: &[f32],
    ) -> Result<Option<TranscriptionResult>> {
        let result = self
            .engine
            .feed_audio(state, samples)
            .map_err(|e| anyhow::anyhow!("Streaming feed failed: {}", e))?;

        Ok(result.map(|r| TranscriptionResult { text: r.text }))
    }

    pub fn finish_streaming(
        &self,
        state: &mut qwen3_asr::StreamingState,
    ) -> Result<TranscriptionResult> {
        let result = self
            .engine
            .finish_streaming(state)
            .map_err(|e| anyhow::anyhow!("Streaming finish failed: {}", e))?;

        Ok(TranscriptionResult { text: result.text })
    }
}
