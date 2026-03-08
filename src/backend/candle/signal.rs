use crate::backend::candle::array::CANDLE_ARRAY;
use candle_core::{Result as CResult, Tensor};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

pub fn hann_window(size: usize) -> CResult<CANDLE_ARRAY> {
    let window: Vec<f32> = (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect();
    Ok(CANDLE_ARRAY::new(Tensor::from_vec(
        window,
        &[size],
        &candle_core::Device::Cpu,
    )?))
}

pub fn reflection_pad1d(
    input: &CANDLE_ARRAY,
    pad_left: usize,
    pad_right: usize,
) -> CResult<CANDLE_ARRAY> {
    let ndim = input.inner.dims().len();
    let dim = ndim - 1;

    // Pad left (reflect)
    let mut padded = if pad_left > 0 {
        let padding = input.inner.narrow(dim, 1, pad_left)?;
        let flipped = padding.flip(&[dim])?;
        Tensor::cat(&[&flipped, &input.inner], dim)?
    } else {
        input.inner.clone()
    };

    if pad_right > 0 {
        let current_len = padded.dims()[dim];
        let start = current_len.saturating_sub(1 + pad_right);
        let padding = padded.narrow(dim, start, pad_right)?;
        let flipped = padding.flip(&[dim])?;
        padded = Tensor::cat(&[&padded, &flipped], dim)?;
    }

    Ok(CANDLE_ARRAY::new(padded))
}

pub fn stft_magnitude(
    input: &CANDLE_ARRAY,
    n_fft: usize,
    hop_length: usize,
    window: &CANDLE_ARRAY,
) -> CResult<CANDLE_ARRAY> {
    // Accurate STFT magnitude using FFT on CPU, then move back to input device.
    let in_dev = input.inner.device().clone();
    let input_cpu = input.inner.to_device(&candle_core::Device::Cpu)?;
    let window_cpu = window.inner.to_device(&candle_core::Device::Cpu)?;
    let signal = input_cpu.to_vec1::<f32>()?;
    let win = window_cpu.to_vec1::<f32>()?;
    let signal_len = signal.len();

    if signal_len < n_fft {
        return Err(candle_core::Error::Msg(format!(
            "Signal length {} is less than FFT size {}",
            signal_len, n_fft
        )));
    }

    let num_frames = (signal_len - n_fft) / hop_length + 1;
    let n_bins = n_fft / 2 + 1;
    let mut magnitudes: Vec<f32> = vec![0.0; num_frames * n_bins];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut frame_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n_fft];

    for t in 0..num_frames {
        let start = t * hop_length;
        for i in 0..n_fft {
            frame_buf[i] = Complex32::new(signal[start + i] * win[i], 0.0);
        }
        fft.process(&mut frame_buf);
        for f in 0..n_bins {
            magnitudes[t * n_bins + f] = frame_buf[f].norm();
        }
    }

    let tensor = Tensor::from_vec(
        magnitudes,
        &[num_frames, n_bins],
        &candle_core::Device::Cpu,
    )?;
    let t = tensor.t()?.to_device(&in_dev)?;
    Ok(CANDLE_ARRAY::new(t))
}

pub fn extract_mel_features(
    waveform: &CANDLE_ARRAY,
    n_fft: usize,
    hop_length: usize,
    _num_mel_bins: usize,
    _sample_rate: u32,
    _fmin: f32,
    _fmax: f32,
) -> CResult<CANDLE_ARRAY> {
    let window = hann_window(n_fft)?;
    let stft_mag = stft_magnitude(waveform, n_fft, hop_length, &window)?;

    // Simplified: return STFT magnitude directly for now
    Ok(stft_mag)
}
