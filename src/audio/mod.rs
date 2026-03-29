use anyhow::Result;

/// Load and resample a WAV file from raw bytes to 16 kHz mono f32 samples.
pub fn process_wav(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.is_empty() {
        return Err("No audio data provided".to_string());
    }

    use std::io::Write;

    let mut temp_file =
        tempfile::NamedTempFile::new().map_err(|e| format!("Failed to create temp file: {}", e))?;

    temp_file
        .write_all(data)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;

    let path = temp_file.path().to_string_lossy().to_string();

    match qwen3_asr::load_audio_wav(&path, 16000) {
        Ok(samples) => {
            if samples.is_empty() {
                Err("No audio samples loaded from WAV file".to_string())
            } else {
                Ok(samples)
            }
        }
        Err(e) => Err(format!("Failed to load audio: {}", e)),
    }
}

/// Simple Energy-based Voice Activity Detection (VAD) and Segmentation.
///
/// Splits a long audio file into smaller chunks (aiming for `max_duration_sec`),
/// by finding the quietest moment near the end of the window to split reliably.
pub fn segment_audio(samples: &[f32], sample_rate: u32, max_duration_sec: f32) -> Vec<Vec<f32>> {
    let total_samples = samples.len();
    let max_len = (max_duration_sec * sample_rate as f32) as usize;

    if total_samples <= max_len {
        return vec![samples.to_vec()];
    }

    let mut segments = Vec::new();
    let mut current_pos = 0;

    // Window size for energy calculation (20 ms)
    let energy_window = (0.02 * sample_rate as f32) as usize;
    // Search for a quiet cut point in the last 5 s of the max_len window
    let search_window = (5.0 * sample_rate as f32) as usize;
    let silence_threshold = 0.005_f32;

    while current_pos < total_samples {
        let remaining = total_samples - current_pos;

        if remaining <= max_len {
            segments.push(samples[current_pos..].to_vec());
            break;
        }

        let search_start = current_pos + max_len.saturating_sub(search_window);
        let search_end = current_pos + max_len;
        let search_start = search_start.max(current_pos);

        let mut best_cut_point = search_end;
        let mut lowest_energy = f32::MAX;

        let mut i = search_end.saturating_sub(energy_window);
        while i >= search_start {
            let window_samples = &samples[i..i + energy_window];
            let energy = calculate_energy(window_samples);

            if energy < lowest_energy {
                lowest_energy = energy;
                best_cut_point = i + energy_window / 2;
            }

            if energy < silence_threshold {
                break;
            }

            i = i.saturating_sub(energy_window / 2);
        }

        segments.push(samples[current_pos..best_cut_point].to_vec());
        current_pos = best_cut_point;
    }

    segments
}

fn calculate_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
