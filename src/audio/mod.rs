use anyhow::Result;

pub struct AudioProcessor;

impl AudioProcessor {
    pub fn new() -> Self {
        Self
    }

    pub fn process_wav(&self, data: &[u8]) -> Result<Vec<f32>, String> {
        if data.is_empty() {
            return Err("No audio data provided".to_string());
        }

        // Use qwen3-asr's built-in WAV loading
        // First, write data to a temporary file
        use std::io::Write;

        let mut temp_file = match tempfile::NamedTempFile::new() {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to create temp file: {}", e)),
        };

        // Write the audio data
        if let Err(e) = temp_file.write_all(data) {
            return Err(format!("Failed to write temp file: {}", e));
        }

        // Get the path
        let path = temp_file.path().to_string_lossy().to_string();

        // Use qwen3-asr's load_audio_wav function
        match qwen3_asr::load_audio_wav(&path, 16000) {
            Ok(samples) => {
                if samples.is_empty() {
                    return Err("No audio samples loaded from WAV file".to_string());
                }
                Ok(samples)
            }
            Err(e) => Err(format!("Failed to load audio: {}", e)),
        }
    }
}

/// Simple Energy-based Voice Activity Detection (VAD) and Segmentation.
/// 
/// Splits a long audio file into smaller chunks (aiming for `max_duration_sec`),
/// by finding the quietest moment near the end of the window to split reliably.
pub fn segment_audio(
    samples: &[f32],
    sample_rate: u32,
    max_duration_sec: f32,
) -> Vec<Vec<f32>> {
    let total_samples = samples.len();
    let max_len = (max_duration_sec * sample_rate as f32) as usize;
    
    // If it's already short enough, return it as a single segment
    if total_samples <= max_len {
        return vec![samples.to_vec()];
    }

    let mut segments = Vec::new();
    let mut current_pos = 0;

    // Window size for energy calculation (e.g., 20ms)
    let energy_window = (0.02 * sample_rate as f32) as usize;
    // We search for a quiet point in the last X seconds of the max_len window
    let search_window = (5.0 * sample_rate as f32) as usize; // search in the last 5s
    // If average energy in a window is below this, we consider it silent enough to cut
    let silence_threshold = 0.005_f32; 

    while current_pos < total_samples {
        let remaining = total_samples - current_pos;
        
        if remaining <= max_len {
            segments.push(samples[current_pos..].to_vec());
            break;
        }

        // We need to cut. Find the best cut point in [current_pos + max_len - search_window, current_pos + max_len]
        let search_start = current_pos + max_len.saturating_sub(search_window);
        let search_end = current_pos + max_len;
        
        // Safety check
        let search_start = search_start.max(current_pos);
        
        let mut best_cut_point = search_end;
        let mut lowest_energy = f32::MAX;

        // Slide window backwards to find the first suitable silence (or the absolute quietest point)
        let mut i = search_end.saturating_sub(energy_window);
        while i >= search_start {
            let window_samples = &samples[i..i + energy_window];
            let energy = calculate_energy(window_samples);
            
            if energy < lowest_energy {
                lowest_energy = energy;
                best_cut_point = i + energy_window / 2; // cut in the middle of the quiet window
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
