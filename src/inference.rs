use anyhow::{Context, Result};
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;
use crate::tensor::{DType, Device, Tensor};

use crate::audio;
use crate::audio_encoder::AudioEncoder;
use crate::config::AsrConfig;
use crate::layers::compute_mrope_cos_sin;
use crate::mel::WhisperFeatureExtractor;
use crate::text_decoder::{create_causal_mask, KvCache, TextDecoder};
use crate::tokenizer::{
    AsrTokenizer, ASR_TEXT_TOKEN_ID, AUDIO_PAD_TOKEN_ID, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID,
};

const MEL_SAMPLE_RATE: u32 = 16000;

/// ASR inference engine.
pub struct AsrInference {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    mel_extractor: WhisperFeatureExtractor,
    tokenizer: AsrTokenizer,
    config: AsrConfig,
    device: Device,
}

impl AsrInference {
    /// Load model from directory containing config.json, model.safetensors, tokenizer.json
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!("Starting model load from {:?}", model_dir);
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        let config: AsrConfig = serde_json::from_str(&config_str)?;

        tracing::info!("Loading weights...");
        let weights = crate::weights::load_model_weights(model_dir, device)?;

        tracing::info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(&weights, "thinker.audio_tower", &config.thinker_config.audio_config, device)?;

        tracing::info!("Loading text decoder...");
        let text_decoder = TextDecoder::load(&weights, "thinker.model", &config.thinker_config.text_config)?;

        tracing::info!("Loading mel extractor...");
        let mel_extractor = WhisperFeatureExtractor::new(
            400, // n_fft
            160, // hop_length
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
            device,
        );

        tracing::info!("Loading tokenizer...");
        let tokenizer =
            AsrTokenizer::from_dir(model_dir).context("Failed to load tokenizer")?;

        tracing::info!("Model loaded successfully");

        Ok(Self {
            audio_encoder,
            text_decoder,
            mel_extractor,
            tokenizer,
            config,
            device,
        })
    }

    /// Transcribe an audio file.
    pub fn transcribe(&self, audio_path: &str, language: Option<&str>) -> Result<TranscribeResult> {
        self.transcribe_with_stream(audio_path, language, None)
    }

    /// Transcribe an audio file with optional partial-text streaming callback channel.
    pub fn transcribe_with_stream(
        &self,
        audio_path: &str,
        language: Option<&str>,
        stream_tx: Option<Sender<String>>,
    ) -> Result<TranscribeResult> {
        crate::tensor::no_grad(|| {
            // Step 1: Load and preprocess audio
            tracing::info!("Loading audio from {}", audio_path);
            let samples = audio::load_audio(audio_path, MEL_SAMPLE_RATE)?;

            // Step 2: Compute mel spectrogram
            let mel = self.mel_extractor.extract(&samples, self.device)?;
            let num_mel_frames = mel.size()[1] as usize;
            tracing::info!("Mel spectrogram: {} frames", num_mel_frames);

            // Step 3: Run audio encoder
            let audio_embeds = self.audio_encoder.forward(&mel);
            let num_audio_tokens = audio_embeds.size()[0] as usize;
            tracing::info!("Audio encoder: {} tokens", num_audio_tokens);

            // Step 4: Build input token sequence
            let (input_ids, audio_positions) =
                self.build_prompt(num_audio_tokens, language)?;
            let seq_len = input_ids.len();

            // Step 5: Build embeddings with audio injection
            let input_tensor =
                Tensor::from_slice_i64(&input_ids).to_device(self.device);
            let mut hidden_states = self.text_decoder.embed(&input_tensor).unsqueeze(0);
            let mut inject_probe_before: Option<f32> = None;
            let mut inject_probe_after: Option<f32> = None;
            let mut inject_audio_norm: Option<f32> = None;

            // Replace audio_pad span with audio encoder embeddings in one shot:
            // [prefix tokens] + [audio embeds] + [suffix tokens]
            if let Some(&audio_start_pos) = audio_positions.first() {
                let before_vec = hidden_states
                    .narrow(1, audio_start_pos as i64, 1)
                    .squeeze_dim(0)
                    .squeeze_dim(0)
                    .to_dtype(DType::Float32)
                    .to_device(Device::Cpu)
                    .to_vec_f32();
                inject_probe_before = Some(
                    before_vec.iter().map(|v| v * v).sum::<f32>().sqrt()
                );
                let audio0_vec = audio_embeds
                    .narrow(0, 0, 1)
                    .squeeze_dim(0)
                    .to_dtype(DType::Float32)
                    .to_device(Device::Cpu)
                    .to_vec_f32();
                inject_audio_norm = Some(
                    audio0_vec.iter().map(|v| v * v).sum::<f32>().sqrt()
                );

                let audio_token_len = audio_positions.len();
                let prefix_len = audio_start_pos;
                let suffix_start = audio_start_pos + audio_token_len;
                let suffix_len = seq_len.saturating_sub(suffix_start);

                let mut segments: Vec<Tensor> = Vec::new();
                if prefix_len > 0 {
                    segments.push(hidden_states.narrow(1, 0, prefix_len as i64));
                }
                segments.push(audio_embeds.unsqueeze(0));
                if suffix_len > 0 {
                    segments.push(hidden_states.narrow(1, suffix_start as i64, suffix_len as i64));
                }
                hidden_states = Tensor::cat(&segments, 1);

                let after_vec = hidden_states
                    .narrow(1, audio_start_pos as i64, 1)
                    .squeeze_dim(0)
                    .squeeze_dim(0)
                    .to_dtype(DType::Float32)
                    .to_device(Device::Cpu)
                    .to_vec_f32();
                inject_probe_after = Some(
                    after_vec.iter().map(|v| v * v).sum::<f32>().sqrt()
                );
            }
            if let (Some(b), Some(a), Some(an)) = (inject_probe_before, inject_probe_after, inject_audio_norm) {
                tracing::info!(
                    "audio_inject_probe: token_hidden_before_l2={:.4} token_hidden_after_l2={:.4} audio_embed_l2={:.4}",
                    b, a, an
                );
            }

            let text_config = &self.config.thinker_config.text_config;
            // Keep decoding bounded on edge devices; allow override for debugging.
            let max_new_tokens = std::env::var("ASR_MAX_NEW_TOKENS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(((num_mel_frames / 2).max(256)).min(1024));

            // Step 6: Build MRoPE position IDs (precompute full table once).
            let position_ids = self.build_position_ids(&input_ids);
            let prefill_len = position_ids[0].len();
            let kv_max_seq = std::env::var("ASR_KV_MAX_SEQ")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4096);
            let reserve = 4usize;
            let max_new_by_kv = kv_max_seq.saturating_sub(prefill_len + reserve);
            let max_new_tokens = if max_new_by_kv == 0 {
                tracing::warn!(
                    "KV budget exhausted by prompt/audio (prefill_len={}, kv_max_seq={}), forcing max_new_tokens=1",
                    prefill_len,
                    kv_max_seq
                );
                1usize
            } else {
                if max_new_tokens > max_new_by_kv {
                    tracing::warn!(
                        "Capping max_new_tokens from {} to {} by KV budget (kv_max_seq={})",
                        max_new_tokens,
                        max_new_by_kv,
                        kv_max_seq
                    );
                }
                max_new_tokens.min(max_new_by_kv)
            };
            let max_seq = prefill_len + max_new_tokens + reserve;
            let full_positions: Vec<i64> = (0..max_seq as i64).collect();
            let full_position_ids: [Vec<i64>; 3] = [
                full_positions.clone(),
                full_positions.clone(),
                full_positions,
            ];
            let (full_cos, full_sin) = compute_mrope_cos_sin(
                &full_position_ids,
                text_config.head_dim,
                text_config.rope_theta,
                &text_config.mrope_section(),
                text_config.mrope_interleaved(),
                self.device,
            );
            let cos = full_cos.narrow(0, 0, seq_len as i64);
            let sin = full_sin.narrow(0, 0, seq_len as i64);

            // Step 7: Prefill
            let mask = create_causal_mask(seq_len as i64, 0, self.device);
            let mut kv_cache = KvCache::new(text_config.num_hidden_layers, max_seq as i64);

            let t_prefill0 = Instant::now();
            let hidden_prefill = self.text_decoder.forward_hidden(
                &hidden_states,
                &cos,
                &sin,
                &mut kv_cache,
                Some(&mask),
            );
            let prefill_ms = t_prefill0.elapsed().as_millis();

            // Step 8: Autoregressive generation
            let mut generated_ids: Vec<i64> = Vec::new();
            let timeout_secs = std::env::var("ASR_DECODE_TIMEOUT_SEC")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(300);
            let decode_deadline = Instant::now() + Duration::from_secs(timeout_secs);
            let mut last_token: Option<i64> = None;
            let mut same_token_run: usize = 0;
            let eos_token_ids = vec![ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];
            let mut seen_asr_text = false;
            let mut text_tokens_after_asr = 0usize;

            let prefill_last_hidden = hidden_prefill.narrow(1, seq_len as i64 - 1, 1);
            let mut next_logits = self
                .text_decoder
                .project_logits(&prefill_last_hidden)
                .squeeze_dim(1);

            let mut current_pos = position_ids[0].len();
            tracing::info!("Starting autoregressive generation...");
            let t_decode0 = Instant::now();
            let debug_probe = std::env::var("ASR_DEBUG_PROBE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let enable_preview = stream_tx.is_some()
                || std::env::var("ASR_LOG_PREVIEW")
                    .ok()
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);

            for i in 0..max_new_tokens {
                let step_t0 = Instant::now();
                if Instant::now() >= decode_deadline {
                    tracing::warn!(
                        "Stopping generation by timeout after {} tokens",
                        generated_ids.len()
                    );
                    break;
                }
                if i > 0 && i % 50 == 0 {
                    tracing::debug!("Generated {} tokens so far...", i);
                }
                let next_token = next_logits.argmax(-1, false).int64_value(&[0]);
                if i == 0 && debug_probe {
                    let logits_vec = next_logits
                        .to_dtype(DType::Float32)
                        .to_device(Device::Cpu)
                        .squeeze_dim(0)
                        .to_vec_f32();
                    let mut max_idx = 0usize;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut nan_count = 0usize;
                    for (idx, &v) in logits_vec.iter().enumerate() {
                        if v.is_nan() {
                            nan_count += 1;
                            continue;
                        }
                        if v > max_val {
                            max_val = v;
                            max_idx = idx;
                        }
                    }
                    tracing::info!(
                        "logits_probe step0: argmax_token={} cpu_max_idx={} cpu_max_val={} nan_count={} vocab={}",
                        next_token,
                        max_idx,
                        max_val,
                        nan_count,
                        logits_vec.len()
                    );
                }
                if i < 30 {
                    tracing::debug!("gen step={} token_id={}", i, next_token);
                }

                if eos_token_ids.contains(&next_token) {
                    // Avoid terminating immediately after "<asr_text>" header; require some body tokens first.
                    if seen_asr_text && text_tokens_after_asr < 4 {
                        tracing::debug!(
                            "Ignoring early EOS token={} (text tokens after <asr_text>: {})",
                            next_token,
                            text_tokens_after_asr
                        );
                    } else {
                        tracing::info!("Generation stopped by EOS at {} tokens", generated_ids.len());
                        break;
                    }
                }

                if Some(next_token) == last_token {
                    same_token_run += 1;
                } else {
                    same_token_run = 0;
                    last_token = Some(next_token);
                }
                if same_token_run >= 128 {
                    tracing::warn!(
                        "Stopping generation by repetition guard at {} tokens (token={})",
                        generated_ids.len(),
                        next_token
                    );
                    break;
                }

                generated_ids.push(next_token);
                if next_token == ASR_TEXT_TOKEN_ID {
                    seen_asr_text = true;
                    text_tokens_after_asr = 0;
                } else if seen_asr_text {
                    text_tokens_after_asr += 1;
                }
                if enable_preview && generated_ids.len() % 5 == 0 {
                    let preview_ids: Vec<i64> = if let Some(pos) =
                        generated_ids.iter().rposition(|&t| t == ASR_TEXT_TOKEN_ID)
                    {
                        generated_ids[pos + 1..].to_vec()
                    } else {
                        generated_ids.clone()
                    };
                    if !preview_ids.is_empty() {
                        if let Ok(preview) = self.tokenizer.decode(&preview_ids) {
                            tracing::info!("gen preview ({} toks): {:?}", generated_ids.len(), preview);
                            if let Some(tx) = stream_tx.as_ref() {
                                let _ = tx.blocking_send(preview);
                            }
                        }
                    }
                }

                let t_embed0 = Instant::now();
                let next_input = Tensor::from_slice_i64(&[next_token]).to_device(self.device);
                let next_hidden = self.text_decoder.embed(&next_input).unsqueeze(0);
                let t_embed = t_embed0.elapsed();

                let new_cos = full_cos.narrow(0, current_pos as i64, 1);
                let new_sin = full_sin.narrow(0, current_pos as i64, 1);

                let t_dec0 = Instant::now();
                let next_hidden_out = self.text_decoder.forward_hidden(
                    &next_hidden,
                    &new_cos,
                    &new_sin,
                    &mut kv_cache,
                    None,
                );
                let t_dec = t_dec0.elapsed();
                let t_lm0 = Instant::now();
                next_logits = self
                    .text_decoder
                    .project_logits(&next_hidden_out)
                    .squeeze_dim(1);
                let t_lm = t_lm0.elapsed();
                tracing::debug!(
                    "gen step={} token={} total_ms={} embed_ms={} dec_ms={} lm_ms={} kv_len={}",
                    i,
                    next_token,
                    step_t0.elapsed().as_millis(),
                    t_embed.as_millis(),
                    t_dec.as_millis(),
                    t_lm.as_millis(),
                    kv_cache.seq_len()
                );

                current_pos += 1;
            }

            // Step 9: Parse output
            let decode_ms = t_decode0.elapsed().as_millis() as f64;
            let tok_per_sec = if decode_ms > 0.0 {
                (generated_ids.len() as f64) / (decode_ms / 1000.0)
            } else {
                0.0
            };
            tracing::info!(
                "decode_perf: prefill_ms={} decode_ms={} tokens={} tok_per_sec={:.3}",
                prefill_ms,
                decode_ms as u64,
                generated_ids.len(),
                tok_per_sec
            );
            tracing::info!("Generated {} tokens", generated_ids.len());
            tracing::info!("Generated token ids: {:?}", generated_ids);
            let content_ids: Vec<i64> = if let Some(pos) =
                generated_ids.iter().rposition(|&t| t == ASR_TEXT_TOKEN_ID)
            {
                generated_ids[pos + 1..].to_vec()
            } else {
                generated_ids.clone()
            };
            let raw_text = self.tokenizer.decode(&content_ids)?;
            let raw_with_special = self.tokenizer.decode_with_special(&generated_ids, false)?;
            tracing::debug!("Raw output: {:?}", raw_text);
            tracing::info!("Raw output with special tokens: {:?}", raw_with_special);
            let (language_detected, transcription) = parse_asr_output(&raw_text, language.is_some());

            Ok(TranscribeResult {
                text: transcription,
                language: language_detected,
                raw_output: raw_text,
            })
        })
    }

    fn build_prompt(
        &self,
        num_audio_tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, Vec<usize>)> {
        let mut tokens: Vec<i64> = vec![
            151644, // <|im_start|>
            8948,   // system
            198,    // \n
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
            872,    // user
            198,    // \n
            151669, // <|audio_start|>
        ];

        let audio_start_pos = tokens.len();
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_PAD_TOKEN_ID);
        }
        let audio_positions: Vec<usize> =
            (audio_start_pos..audio_start_pos + num_audio_tokens).collect();

        tokens.extend_from_slice(&[
            151670, // <|audio_end|>
        ]);

        // Explicit ASR instruction is critical; without it the model tends to emit only template headers.
        let instruction = if let Some(lang) = language {
            format!(
                "\nPlease transcribe the speech audio into {}. Return only recognized text.",
                capitalize_first(lang)
            )
        } else {
            "\nPlease transcribe the speech audio. Return only recognized text.".to_string()
        };
        tokens.extend(self.tokenizer.encode(&instruction)?);

        tokens.extend_from_slice(&[
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
        ]);

        if let Some(lang) = language {
            tokens.push(77091); // assistant
            tokens.push(198);   // \n
            let prefix = format!("language {}", capitalize_first(lang));
            tokens.extend(self.tokenizer.encode(&prefix)?);
        } else {
            tokens.push(77091); // assistant
            tokens.push(198);   // \n
        }

        Ok((tokens, audio_positions))
    }

    fn build_position_ids(
        &self,
        input_ids: &[i64],
    ) -> [Vec<i64>; 3] {
        let seq_len = input_ids.len();
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        [positions.clone(), positions.clone(), positions]
    }
}

/// Result of ASR transcription.
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

fn parse_asr_output(raw: &str, language_forced: bool) -> (String, String) {
    if language_forced {
        return ("forced".to_string(), raw.trim().to_string());
    }

    let raw = raw.trim();

    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(asr_pos) = rest.find("<asr_text>") {
            let lang = rest[..asr_pos].trim().to_string();
            let text = rest[asr_pos + "<asr_text>".len()..].trim().to_string();
            return (lang, text);
        }
        let mut lang_end = 0;
        for (i, c) in rest.char_indices() {
            if c.is_whitespace() || !c.is_alphabetic() {
                lang_end = i;
                break;
            }
            lang_end = i + c.len_utf8();
        }
        if lang_end > 0 {
            let lang = rest[..lang_end].to_string();
            let text = rest[lang_end..].trim().to_string();
            return (lang, text);
        }
    }

    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
