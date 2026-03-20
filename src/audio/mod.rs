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
