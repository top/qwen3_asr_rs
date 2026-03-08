use anyhow::{Context, Result};

/// Load a WAV file and resample to target rate using rubato.
pub fn load_audio(path: &str, target_sample_rate: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path).context("Failed to open WAV file")?;
    let spec = reader.spec();

    tracing::info!(
        "WAV: {}ch, {}Hz, {:?}, {}bit",
        spec.channels,
        spec.sample_rate,
        spec.sample_format,
        spec.bits_per_sample
    );

    // Read all samples as f32
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .map(|s| s.unwrap())
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples = if spec.channels > 1 {
        let channels = spec.channels as usize;
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample if needed
    let samples = if spec.sample_rate != target_sample_rate {
        resample(&mono_samples, spec.sample_rate, target_sample_rate)?
    } else {
        mono_samples
    };

    tracing::info!(
        "Loaded WAV: {} samples ({:.2}s at {}Hz)",
        samples.len(),
        samples.len() as f64 / target_sample_rate as f64,
        target_sample_rate
    );

    Ok(samples)
}

/// Resample audio using rubato.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, Resampler, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1, // mono
    )
    .map_err(|e| anyhow::anyhow!("Failed to create resampler: {}", e))?;

    let result = resampler
        .process(&[samples.to_vec()], None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?;

    Ok(result.into_iter().next().unwrap_or_default())
}
