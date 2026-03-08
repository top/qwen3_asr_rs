//! ONNX Runtime backend implementation for Qwen3 ASR
//!
//! This module provides a unified Tensor API that wraps the onnxruntime crate.

use anyhow::{Context, Result};
use ndarray::{Array, ArrayD};
use std::path::Path;

// Re-export onnxruntime types for external use
pub use onnxruntime::*;

// ---------------------------------------------------------------------------
// Device — compute device abstraction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

impl Device {
    pub fn gpu() -> Self {
        Device::Gpu(0)
    }

    #[allow(dead_code)]
    pub fn new_cuda(device_id: usize) -> Self {
        Device::Gpu(device_id)
    }
}

// ---------------------------------------------------------------------------
// DType — data type abstraction (matches ONNX Runtime types)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Bool,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Int64 => "int64",
            DType::Int32 => "int32",
            DType::Bool => "bool",
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor — unified tensor type using ONNX Runtime OrtValue
// ---------------------------------------------------------------------------

pub struct Tensor {
    // Store data in Vec<f32> for simplicity; shape and dtype tracked separately
    data: Vec<f32>,
    shape: Vec<i64>,
    dtype: DType,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={})", self.shape, self.dtype.as_str())
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl Tensor {
    // -- Creation --

    pub fn zeros(shape: &[i64], dtype: DType, _device: &Device) -> Self {
        let size = shape.iter().product::<i64>() as usize;
        Tensor {
            data: vec![0.0f32; size],
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn ones(shape: &[i64], dtype: DType, _device: &Device) -> Self {
        let size = shape.iter().product::<i64>() as usize;
        Tensor {
            data: vec![1.0f32; size],
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, _device: &Device) -> Self {
        let size = shape.iter().product::<i64>() as usize;
        Tensor {
            data: vec![val as f32; size],
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn arange(start: i64, end: i64, _device: &Device) -> Self {
        let len = (end - start) as usize;
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        Tensor {
            data,
            shape: vec![len as i64],
            dtype: DType::Int64.into(), // Simplified
        }
    }

    pub fn arange_f(start: f64, end: f64, _device: &Device) -> Self {
        let len = ((end - start) / 1.0) as usize;
        let data: Vec<f32> = (start..end).step_by(1).map(|x| x as f32).collect();
        Tensor {
            data,
            shape: vec![len as i64],
            dtype: DType::Float32,
        }
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        // Concatenate tensors along specified dimension
        let mut result_data = Vec::new();
        for tensor in tensors {
            result_data.extend(tensor.data.clone());
        }
        
        let mut new_shape = tensor.shape.clone();
        if !tensors.is_empty() {
            let dim_idx = dim as usize;
            let total: i64 = tensors.iter().map(|t| t.shape[dim_idx]).sum();
            new_shape[dim_idx] = total;
        }

        Tensor {
            data: result_data,
            shape: new_shape,
            dtype: if !tensors.is_empty() { tensors[0].dtype } else { DType::Float32 },
        }
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        // Stack tensors along a new dimension
        let mut result_data = Vec::new();
        for tensor in tensors {
            result_data.extend(tensor.data.clone());
        }

        let mut new_shape = tensors[0].shape.clone();
        new_shape.insert(dim as usize, tensors.len() as i64);

        Tensor {
            data: result_data,
            shape: new_shape,
            dtype: tensors[0].dtype,
        }
    }

    // -- Shape operations --

    pub fn size(&self) -> Vec<i64> {
        self.shape.clone()
    }

    pub fn size3(&self) -> (i64, i64, i64) {
        let s = &self.shape;
        if s.len() >= 3 {
            (s[0], s[1], s[2])
        } else {
            (0, 0, 0)
        }
    }

    pub fn size4(&self) -> (i64, i64, i64, i64) {
        let s = &self.shape;
        if s.len() >= 4 {
            (s[0], s[1], s[2], s[3])
        } else {
            (0, 0, 0, 0)
        }
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        let size = shape.iter().product::<i64>() as usize;
        Tensor {
            data: self.data.clone(), // In real impl, would actually reshape
            shape: shape.to_vec(),
            dtype: self.dtype,
        }
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        let mut result_data = Vec::new();
        // Simplified: just return clone for now
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim as usize, 1);
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        }
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        let mut new_shape = self.shape.clone();
        if dim as usize < new_shape.len() && new_shape[dim as usize] == 1 {
            new_shape.remove(dim as usize);
        }
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        }
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let mut new_shape = self.shape.clone();
        if dim0 as usize < new_shape.len() && dim1 as usize < new_shape.len() {
            new_shape.swap(dim0 as usize, dim1 as usize);
        }
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        }
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        let mut new_shape = Vec::new();
        for &d in dims {
            if d as usize < self.shape.len() {
                new_shape.push(self.shape[d as usize]);
            }
        }
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        }
    }

    pub fn expand(&self, size: &[i64], _implicit: bool) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: size.to_vec(),
            dtype: self.dtype,
        }
    }

    pub fn contiguous(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    pub fn tr(&self) -> Self {
        let mut new_shape = self.shape.clone();
        if new_shape.len() >= 2 {
            new_shape.swap(new_shape.len() - 2, new_shape.len() - 1);
        }
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        }
    }

    pub fn get(&self, index: i64) -> Self {
        // Simplified: return clone for now
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        // Simplified: return clone for now
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    // -- Arithmetic operations --

    pub fn matmul(&self, other: &Self) -> Self {
        // Matrix multiplication: self (m x k) * other (k x n) = result (m x n)
        let m = self.shape[0] as usize;
        let k = self.shape[1] as usize;
        let n = other.shape[1] as usize;
        
        // Validate shapes match for matrix multiplication
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Tensor { data: vec![], shape: vec![0], dtype: DType::Float32 };
        }
        
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    if i * k + l < self.data.len() && l * n + j < other.data.len() {
                        sum += self.data[i * k + l] * other.data[l * n + j];
                    }
                }
                result[i * n + j] = sum;
            }
        }
        
        Tensor { data: result, shape: vec![m as i64, n as i64], dtype: DType::Float32 }
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.powf(exp)).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn neg(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(min as f32)).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn maximum(&self, other: &Self) -> Self {
        let data: Vec<f32> = self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a.max(b))
            .collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn where_cond(&self, condition: &Self, other: &Self) -> Self {
        let data: Vec<f32> = self.data.iter().zip(other.data.iter())
            .zip(condition.data.iter())
            .map(|((a, b), &cond)| if cond != 0.0 { *a } else { *b })
            .collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    // -- Math operations --

    pub fn abs(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.abs()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn square(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x * x).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn sqrt(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn rsqrt(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| 1.0 / (x + 1e-8)).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn log10(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.log10()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn sin(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.sin()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn cos(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.cos()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn exp(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x.exp()).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    // -- Activations --

    pub fn softmax(&self, _dim: i64) -> Self {
        // Simplified placeholder
        Tensor {
            data: vec![0.0f32; self.data.len()],
            shape: self.shape.clone(),
            dtype: DType::Float32,
        }
    }

    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x * (1.0 + (-x*x).exp()).erf() / 2.0).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn silu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    // -- Reduction operations --

    pub fn mean_dim(&self, _dims: &[i64], _keepdim: bool) -> Self {
        let mean_val = self.data.iter().sum::<f32>() / self.data.len() as f32;
        Tensor { data: vec![mean_val], shape: vec![], dtype: DType::Float32 }
    }

    pub fn max(&self) -> Self {
        let max_val = *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        Tensor { data: vec![max_val], shape: vec![], dtype: DType::Float32 }
    }

    // -- Indexing operations --

    pub fn argmax(&self, _dim: i64, _keepdim: bool) -> Self {
        let max_idx = self.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap_or(0);
        Tensor { data: vec![max_idx as f32], shape: vec![], dtype: DType::Int64 }
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        // Simplified placeholder - would need proper matrix handling for real trilu
        Tensor { data: self.data.clone(), shape: self.shape.clone(), dtype: self.dtype }
    }

    // -- Type / Device operations --

    pub fn to_dtype(&self, _dtype: DType) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: _dtype,
        }
    }

    pub fn to_device(&self, _device: &Device) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    pub fn kind(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        static CPU_DEVICE: Device = Device::Cpu;
        &CPU_DEVICE
    }

    pub fn shallow_clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    // -- Data extraction --

    pub fn int64_value(&self, _indices: &[i64]) -> i64 {
        self.data[0] as i64
    }

    pub fn f64_value(&self, _indices: &[i64]) -> f64 {
        self.data[0] as f64
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.data.clone()
    }

    // -- Convolution (placeholder for ONNX Runtime model loading) --

    pub fn conv2d(
        &self,
        _weight: &Self,
        _bias: Option<&Self>,
        _stride: &[i64],
        _padding: &[i64],
        _dilation: &[i64],
        _groups: i64,
    ) -> Self {
        // Placeholder - real convolution would require ONNX Runtime session execution
        Tensor {
            data: vec![0.0f32; 10],
            shape: vec![1, 10],
            dtype: DType::Float32,
        }
    }

    // -- Signal operations (placeholder) --

    pub fn reflection_pad1d(&self, _pad: &[i64]) -> Self {
        Tensor { data: self.data.clone(), shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn stft(
        &self,
        _n_fft: i64,
        _hop_length: i64,
        _win_length: i64,
        _window: &Self,
        _normalized: bool,
        _onesided: bool,
        _return_complex: bool,
    ) -> Self {
        Tensor { data: self.data.clone(), shape: self.shape.clone(), dtype: self.dtype }
    }

    pub fn hann_window(_size: i64, _device: &Device) -> Self {
        let size = _size as usize;
        let mut window = vec![0.0f32; size];
        for i in 0..size {
            // Hann window formula
            window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (_size as f32)).cos());
        }
        Tensor { data: window, shape: vec![_size], dtype: DType::Float32 }
    }

    pub fn embedding(_weight: &Self, _indices: &Self) -> Self {
        Tensor { data: vec![0.0f32; 10], shape: vec![10], dtype: DType::Float32 }
    }
}

// ONNX Runtime Session wrapper for model loading and inference
pub struct OnnxSession {
    pub environment: Environment,
    pub session: onnxruntime::session::Session,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OnnxSession(initialized={})", self.initialized())
    }
}

impl OnnxSession {
    pub fn load(model_path: &Path) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("qwen3-asr")
            .build()
            .context("Failed to build ONNX Runtime environment")?;

        let session = environment
            .new_session_builder()
            .context("Failed to create session builder")?
            .with_model_from_file(model_path)
            .context("Failed to load ONNX model from file")?;

        Ok(OnnxSession { environment, session })
    }

    pub fn initialized(&self) -> bool {
        true // Session is always valid after successful construction
    }

    pub fn run(&self, inputs: &[&Tensor], input_names: &[&str]) -> Result<Vec<Tensor>> {
        // Convert our Tensor inputs to onnxruntime OrtOwnedTensor
        let ort_inputs: Vec<OrtOwnedTensor<f32, _>> = inputs
            .iter()
            .zip(input_names.iter())
            .map(|(tensor, name)| {
                let shape: &[i64] = &tensor.shape;
                // Flatten the data for onnxruntime
                let array = Array::from_shape_vec(shape.to_vec(), tensor.data.clone())
                    .context("Failed to reshape input tensor")?;
                Ok(OrtOwnedTensor::from_owned_tensor(array))
            })
            .collect::<Result<Vec<_>>>()?;

        // Run the model and get outputs
        let ort_outputs = self.session.run(&ort_inputs)
            .context("Failed to run ONNX session")?;

        // Convert OrtOwnedTensor back to our Tensor type
        let results: Result<Vec<Tensor>> = ort_outputs
            .into_iter()
            .map(|ort_tensor| {
                let shape = ort_tensor.shape();
                let data = ort_tensor.to_vec().context("Failed to extract tensor data")?;
                
                // Determine dtype - always float32 for now
                Tensor {
                    data,
                    shape: shape.iter().map(|&s| s as i64).collect(),
                    dtype: DType::Float32,
                }
            })
            .collect();

        results
    }

    pub fn input_names(&self) -> Result<Vec<String>> {
        self.session.input_names()
            .context("Failed to get input names")
            .map(|names| names.iter().map(|s| s.to_string()).collect())
    }

    pub fn output_names(&self) -> Result<Vec<String>> {
        self.session.output_names()
            .context("Failed to get output names")
            .map(|names| names.iter().map(|s| s.to_string()).collect())
    }
}

// Re-export types for backend module
pub use Device;
pub use DType;
pub use Tensor;
