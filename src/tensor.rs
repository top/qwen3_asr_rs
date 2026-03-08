//! Unified tensor abstraction over candle (CUDA) and MLX backends.
use crate::backend::candle::array::CANDLE_ARRAY;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Bool,
}

#[cfg(feature = "candle-backend")]
impl From<DType> for candle_core::DType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float32 => candle_core::DType::F32,
            DType::Float16 => candle_core::DType::F16,
            DType::BFloat16 => candle_core::DType::BF16,
            DType::Int64 => candle_core::DType::I64,
            DType::Int32 => candle_core::DType::I32,
            DType::Bool => candle_core::DType::U8,
        }
    }
}

#[cfg(feature = "candle-backend")]
impl From<&candle_core::DType> for DType {
    fn from(dt: &candle_core::DType) -> Self {
        match dt {
            candle_core::DType::F32 => DType::Float32,
            candle_core::DType::F16 => DType::Float16,
            candle_core::DType::BF16 => DType::BFloat16,
            candle_core::DType::I64 => DType::Int64,
            candle_core::DType::I32 => DType::Int32,
            _ => DType::Float32,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<DType> for crate::backend::mlx::ffi::mlx_dtype {
    fn from(dt: DType) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            DType::Float32 => MLX_FLOAT32,
            DType::Float16 => MLX_FLOAT16,
            DType::BFloat16 => MLX_BFLOAT16,
            DType::Int64 => MLX_INT64,
            DType::Int32 => MLX_INT32,
            DType::Bool => MLX_BOOL,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<crate::backend::mlx::ffi::mlx_dtype> for DType {
    fn from(dt: crate::backend::mlx::ffi::mlx_dtype) -> Self {
        use crate::backend::mlx::ffi::mlx_dtype::*;
        match dt {
            MLX_FLOAT32 | MLX_FLOAT64 => DType::Float32,
            MLX_FLOAT16 => DType::Float16,
            MLX_BFLOAT16 => DType::BFloat16,
            MLX_INT64 => DType::Int64,
            MLX_INT32 | MLX_INT8 | MLX_INT16 => DType::Int32,
            MLX_BOOL => DType::Bool,
            _ => DType::Float32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

impl Device {
    pub fn gpu() -> Self {
        Device::Gpu(0)
    }

    #[cfg(feature = "candle-backend")]
    pub fn infer_device() -> Self {
        crate::backend::candle::ffi::infer_device()
    }
}

pub struct Tensor {
    #[cfg(feature = "candle-backend")]
    pub(crate) inner: crate::backend::candle::array::CANDLE_ARRAY,

    #[cfg(feature = "mlx")]
    pub(crate) inner: crate::backend::mlx::array::MlxArray,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?})",
            self.size(),
            self.kind()
        )
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "candle-backend")]
        {
            Tensor {
                inner: self.inner.clone(),
            }
        }

        #[cfg(feature = "mlx")]
        {
            Tensor {
                inner: self.inner.clone(),
            }
        }
    }
}

pub fn no_grad<T, F: FnOnce() -> T>(f: F) -> T {
    f() // Candle and MLX don't construct graphs eagerly
}

#[cfg(feature = "candle-backend")]
impl Tensor {
    pub fn from_candle(a: crate::backend::candle::array::CANDLE_ARRAY) -> Self {
        Tensor { inner: a }
    }

    pub fn as_candle(&self) -> &crate::backend::candle::array::CANDLE_ARRAY {
        &self.inner
    }

    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i64];
        Tensor::from_candle(crate::backend::candle::ops::from_f32(data, &shape).unwrap())
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        let shape = [data.len() as i64];
        Tensor::from_candle(crate::backend::candle::ops::from_i64(data, &shape).unwrap())
    }

    pub fn zeros(shape: &[i64], dtype: DType, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::zeros(shape, dtype).unwrap()).to_device(device)
    }

    pub fn ones(shape: &[i64], dtype: DType, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::ones(shape, dtype).unwrap()).to_device(device)
    }

    pub fn full(shape: &[i64], val: f32, dtype: DType, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::full(shape, val, dtype).unwrap()).to_device(device)
    }

    pub fn arange(start: i64, end: i64, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::arange(start, end).unwrap()).to_device(device)
    }

    pub fn arange_f(start: f32, end: f32, step: f32, dtype: DType, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::arange_f(start, end, step, dtype).unwrap()).to_device(device)
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        let rank = tensors[0].inner.inner.dims().len() as i64;
        let d = if dim < 0 { rank + dim } else { dim };
        let d = d as usize;
        let base_dev = tensors[0].inner.inner.device().clone();
        let base_dtype = tensors[0].inner.inner.dtype();
        let aligned: Vec<CANDLE_ARRAY> = tensors
            .iter()
            .map(|t| {
                let on_dev = t.inner.inner.to_device(&base_dev).unwrap();
                let on_dtype = if on_dev.dtype() != base_dtype {
                    on_dev.to_dtype(base_dtype).unwrap()
                } else {
                    on_dev
                };
                CANDLE_ARRAY::new(on_dtype)
            })
            .collect();
        let refs: Vec<&crate::backend::candle::array::CANDLE_ARRAY> = aligned.iter().collect();
        Tensor::from_candle(crate::backend::candle::ops::concatenate(&refs, d).unwrap())
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        assert!(!tensors.is_empty(), "stack requires at least one tensor");
        let rank = tensors[0].inner.inner.dims().len() as i64 + 1;
        let d = if dim < 0 { rank + dim } else { dim };
        let d = d as usize;
        let base_dev = tensors[0].inner.inner.device().clone();
        let base_dtype = tensors[0].inner.inner.dtype();
        let aligned: Vec<CANDLE_ARRAY> = tensors
            .iter()
            .map(|t| {
                let on_dev = t.inner.inner.to_device(&base_dev).unwrap();
                let on_dtype = if on_dev.dtype() != base_dtype {
                    on_dev.to_dtype(base_dtype).unwrap()
                } else {
                    on_dev
                };
                CANDLE_ARRAY::new(on_dtype)
            })
            .collect();
        let refs: Vec<&crate::backend::candle::array::CANDLE_ARRAY> = aligned.iter().collect();
        Tensor::from_candle(crate::backend::candle::ops::stack(&refs, d).unwrap())
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        let flat_indices = indices.inner.inner.flatten_all().unwrap();
        let ids_u32 = flat_indices.to_dtype(candle_core::DType::U32).unwrap();
        let out = weight.inner.inner.embedding(&ids_u32).unwrap();
        let expected_shape: Vec<usize> = indices
            .inner
            .inner
            .dims()
            .iter()
            .copied()
            .chain(std::iter::once(weight.inner.inner.dims()[1]))
            .collect();
        let reshaped = out.reshape(expected_shape).unwrap();
        Tensor::from_candle(CANDLE_ARRAY::new(reshaped))
    }

    pub fn hann_window(size: i64, device: Device) -> Self {
        Tensor::from_candle(crate::backend::candle::signal::hann_window(size as usize).unwrap()).to_device(device)
    }

    pub fn size(&self) -> Vec<i64> {
        self.inner
            .inner
            .shape()
            .dims()
            .iter()
            .map(|&d| d as i64)
            .collect()
    }

    pub fn size3(&self) -> (i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2])
    }

    pub fn size4(&self) -> (i64, i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2], s[3])
    }

    pub fn dim(&self) -> usize {
        self.inner.inner.dims().len()
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::reshape(&self.inner, shape).unwrap())
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        self.view(shape)
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(
            crate::backend::candle::ops::narrow(&self.inner, d, start as usize, len as usize)
                .unwrap(),
        )
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::unsqueeze(&self.inner, d).unwrap())
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::squeeze_dim(&self.inner, d).unwrap())
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let d0 = if dim0 < 0 {
            (self.dim() as i64 + dim0) as usize
        } else {
            dim0 as usize
        };
        let d1 = if dim1 < 0 {
            (self.dim() as i64 + dim1) as usize
        } else {
            dim1 as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::transpose(&self.inner, d0, d1).unwrap())
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        let dims_usize: Vec<usize> = dims
            .iter()
            .map(|&d| {
                if d < 0 {
                    (self.dim() as i64 + d) as usize
                } else {
                    d as usize
                }
            })
            .collect();
        Tensor::from_candle(crate::backend::candle::ops::permute(&self.inner, &dims_usize).unwrap())
    }

    pub fn expand(&self, size: &[i64], _implicit: bool) -> Self {
        let current = self.inner.inner.dims();
        let target: Vec<usize> = size
            .iter()
            .enumerate()
            .map(|(i, &s)| if s == -1 { current[i] } else { s as usize })
            .collect();
        let out = self.inner.inner.broadcast_as(target).unwrap();
        Tensor::from_candle(CANDLE_ARRAY::new(out))
    }
    pub fn contiguous(&self) -> Self {
        let out = self.inner.inner.contiguous().unwrap();
        Tensor::from_candle(CANDLE_ARRAY::new(out))
    }
    pub fn tr(&self) -> Self {
        self.transpose(-2, -1)
    }

    pub fn get(&self, index: i64) -> Self {
        let idx = if index < 0 {
            (self.size()[0] as i64 + index) as usize
        } else {
            index as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::get(&self.inner, idx).unwrap())
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(
            crate::backend::candle::ops::narrow(&self.inner, d, index as usize, 1).unwrap(),
        )
        .squeeze_dim(dim)
    }

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::matmul(&self.inner, &other.inner).unwrap())
    }

    pub fn pow_scalar(&self, exp: f32) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::pow_scalar(&self.inner, exp).unwrap())
    }

    pub fn neg(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::neg(&self.inner).unwrap())
    }

    pub fn clamp_min(&self, min: f32) -> Self {
        let min_arr = crate::backend::candle::array::CANDLE_ARRAY::new(
            (candle_core::Tensor::ones_like(&self.inner.inner).unwrap() * (min as f64)).unwrap(),
        );
        Tensor::from_candle(crate::backend::candle::ops::maximum(&self.inner, &min_arr).unwrap())
    }

    pub fn maximum(&self, other: &Tensor) -> Self {
        let lhs = self.inner.inner.clone();
        let rhs_on_lhs = other.inner.inner.to_device(lhs.device()).unwrap();
        let rhs = if rhs_on_lhs.dtype() != lhs.dtype() {
            rhs_on_lhs.to_dtype(lhs.dtype()).unwrap()
        } else {
            rhs_on_lhs
        };
        let rhs_arr = CANDLE_ARRAY::new(rhs.clone());

        if let Ok(result) = crate::backend::candle::ops::maximum(&CANDLE_ARRAY::new(lhs.clone()), &rhs_arr) {
            return Tensor::from_candle(result);
        }

        let self_shape = lhs.shape().dims();
        let other_shape = rhs.shape().dims();
        let self_numel: usize = self_shape.iter().product();
        let other_numel: usize = other_shape.iter().product();

        if other_numel == 1 && self_numel > 1 {
            let scalar = rhs.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let other_expanded = CANDLE_ARRAY::new(
                (candle_core::Tensor::ones_like(&lhs).unwrap() * (scalar as f64))
                    .unwrap(),
            );
            return Tensor::from_candle(
                crate::backend::candle::ops::maximum(&CANDLE_ARRAY::new(lhs), &other_expanded).unwrap(),
            );
        }

        if self_numel == 1 && other_numel > 1 {
            let scalar = lhs.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
            let self_expanded = CANDLE_ARRAY::new(
                (candle_core::Tensor::ones_like(&rhs).unwrap() * (scalar as f64))
                    .unwrap(),
            );
            return Tensor::from_candle(
                crate::backend::candle::ops::maximum(&self_expanded, &rhs_arr).unwrap(),
            );
        }

        Tensor::from_candle(crate::backend::candle::ops::maximum(&CANDLE_ARRAY::new(lhs), &rhs_arr).unwrap())
    }

    pub fn abs(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::abs(&self.inner).unwrap())
    }

    pub fn square(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::square(&self.inner).unwrap())
    }

    pub fn sqrt(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::sqrt(&self.inner).unwrap())
    }

    pub fn rsqrt(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::rsqrt(&self.inner).unwrap())
    }

    pub fn log10(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::log10(&self.inner).unwrap())
    }

    pub fn sin(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::sin(&self.inner).unwrap())
    }

    pub fn cos(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::cos(&self.inner).unwrap())
    }

    pub fn exp(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::exp(&self.inner).unwrap())
    }

    pub fn softmax(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::softmax(&self.inner, d).unwrap())
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::gelu(&self.inner).unwrap())
    }

    pub fn silu(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::silu(&self.inner).unwrap())
    }

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        let dims_usize: Vec<usize> = dims
            .iter()
            .map(|&d| {
                if d < 0 {
                    (self.dim() as i64 + d) as usize
                } else {
                    d as usize
                }
            })
            .collect();
        Tensor::from_candle(
            crate::backend::candle::ops::mean(&self.inner, &dims_usize, keepdim).unwrap(),
        )
    }

    pub fn max(&self) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::max(&self.inner).unwrap())
    }

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        Tensor::from_candle(crate::backend::candle::ops::argmax(&self.inner, d, keepdim).unwrap())
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        let result = crate::backend::candle::ops::triu(&self.inner, diagonal as i32).unwrap();
        Tensor::from_candle(result)
    }

    pub fn slice_scatter(&self, src: &Tensor, dim: i64, start: i64, end: i64, _step: i64) -> Self {
        let d = if dim < 0 {
            (self.dim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        let dim_len = self.size()[d] as i64;
        let end = end.clamp(0, dim_len);
        let mut parts: Vec<Tensor> = Vec::new();
        if start > 0 {
            parts.push(self.narrow(dim, 0, start));
        }
        parts.push(src.clone());
        if end < dim_len {
            parts.push(self.narrow(dim, end, dim_len - end));
        }
        Tensor::cat(&parts, dim)
    }

    pub fn fill_(&self, val: f32) -> Self {
        let shape = self.size();
        Tensor::full(&shape, val, DType::Float32, self.device())
    }

    pub fn layer_norm(
        &self,
        _normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Self {
        // Simplified layer norm - compute mean/var manually
        let x = &self.inner.inner;
        let ndim = x.dims().len();
        let last_dim = x.dims()[ndim - 1];

        // Normalize along last dimension
        let x_f32 = x.to_dtype(candle_core::DType::F32).unwrap();
        let sum = x_f32.sum_keepdim(ndim - 1).unwrap();
        let mean_val = (&sum / (last_dim as f64)).unwrap();

        // Broadcast mean to full shape
        let mean_shape: Vec<usize> = x.dims().to_vec();
        let mean_tensor = mean_val.broadcast_as(mean_shape.as_slice()).unwrap();
        let diff = (&x_f32 - &mean_tensor).unwrap();

        // Compute variance
        let var = (&diff * &diff).unwrap().sum_keepdim(ndim - 1).unwrap() / (last_dim as f64);
        let var_tensor = var.unwrap();
        let inv_std = (var_tensor + eps as f64).unwrap().sqrt().unwrap().recip().unwrap();
        let inv_std = inv_std.broadcast_as(mean_shape.as_slice()).unwrap();
        let norm = (&diff * &inv_std).unwrap();

        // Apply weight and bias if provided
        let mut result = norm.to_dtype(x.dtype()).unwrap();
        if let Some(w) = weight {
            let w_on_result = w.inner.inner.to_device(result.device()).unwrap();
            let w_aligned = if result.dtype() != w_on_result.dtype() {
                w_on_result.to_dtype(result.dtype()).unwrap()
            } else {
                w_on_result
            };
            result = (&result * &w_aligned)
                .or_else(|_| result.broadcast_mul(&w_aligned))
                .or_else(|_| w_aligned.broadcast_mul(&result))
                .unwrap();
        }
        if let Some(b) = bias {
            let b_on_result = b.inner.inner.to_device(result.device()).unwrap();
            let b_aligned = if result.dtype() != b_on_result.dtype() {
                b_on_result.to_dtype(result.dtype()).unwrap()
            } else {
                b_on_result
            };
            result = (&result + &b_aligned)
                .or_else(|_| result.broadcast_add(&b_aligned))
                .or_else(|_| b_aligned.broadcast_add(&result))
                .unwrap();
        }

        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }

    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        _dilation: &[i64],
        _groups: i64,
    ) -> Self {
        let p = padding[0] as usize;
        let s = stride[0] as usize;
        let w_dev = weight.inner.inner.device().clone();
        let x_src = self.inner.inner.to_device(&w_dev).unwrap();
        let w_src = weight.inner.inner.clone();

        let x_dtype = x_src.dtype();
        let w_dtype = w_src.dtype();
        let (x_for_conv, w_for_conv) = if x_dtype != w_dtype {
            tracing::warn!(
                "candle conv2d dtype mismatch (x={:?}, w={:?}), normalizing both to F32",
                x_dtype,
                w_dtype
            );
            (
                x_src.to_dtype(candle_core::DType::F32).unwrap(),
                w_src.to_dtype(candle_core::DType::F32).unwrap(),
            )
        } else {
            (x_src, w_src)
        };

        let out = match x_for_conv.conv2d(&w_for_conv, p, s, 1, 1) {
            Ok(v) => v,
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("unsupported dtype BF16") {
                    tracing::warn!(
                        "candle conv2d BF16 unsupported, falling back to F32 (x={:?}, w={:?})",
                        x_for_conv.shape().dims(),
                        w_for_conv.shape().dims()
                    );
                    let x_f32 = x_for_conv.to_dtype(candle_core::DType::F32).unwrap();
                    let w_f32 = w_for_conv.to_dtype(candle_core::DType::F32).unwrap();
                    x_f32.conv2d(&w_f32, p, s, 1, 1).unwrap()
                } else {
                    panic!("conv2d failed: {}", err);
                }
            }
        };
        
        let result = if let Some(b) = bias {
            let b_shape = b.inner.inner.shape().dims();
            let b_reshaped = b
                .inner
                .inner
                .to_dtype(out.dtype())
                .unwrap()
                .reshape((1, b_shape[0], 1, 1))
                .unwrap();
            out.broadcast_add(&b_reshaped).unwrap()
        } else {
            out
        };
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_candle(
            crate::backend::candle::signal::reflection_pad1d(
                &self.inner,
                pad[0] as usize,
                pad[1] as usize,
            )
            .unwrap(),
        )
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        _win_length: i64,
        window: &Tensor,
        _normalized: bool,
        _onesided: bool,
    ) -> Self {
        Tensor::from_candle(
            crate::backend::candle::signal::stft_magnitude(
                &self.inner,
                n_fft as usize,
                hop_length as usize,
                &window.inner,
            )
            .unwrap(),
        )
    }

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_candle(crate::backend::candle::ops::to_dtype(&self.inner, dtype).unwrap())
    }

    pub fn to_device(&self, device: Device) -> Self {
        let target = crate::backend::candle::ffi::device_to_candle(device);
        Tensor::from_candle(CANDLE_ARRAY::new(self.inner.inner.to_device(&target).unwrap()))
    }
    pub fn kind(&self) -> DType {
        (&self.inner.inner.dtype()).into()
    }
    pub fn device(&self) -> Device {
        match self.inner.inner.device().location() {
            candle_core::DeviceLocation::Cpu => Device::Cpu,
            candle_core::DeviceLocation::Cuda { gpu_id } => Device::Gpu(gpu_id),
            candle_core::DeviceLocation::Metal { gpu_id } => Device::Gpu(gpu_id),
        }
    }
    pub fn shallow_clone(&self) -> Self {
        self.clone()
    }

    pub fn int64_value(&self, indices: &[i64]) -> i64 {
        let flat = self
            .inner
            .inner
            .flatten_all()
            .unwrap()
            .to_device(&candle_core::Device::Cpu)
            .unwrap();
        if let Ok(v) = flat.to_vec1::<i64>() {
            return v[indices[0] as usize];
        }
        if let Ok(v) = flat.to_vec1::<u32>() {
            return v[indices[0] as usize] as i64;
        }
        if let Ok(v) = flat.to_vec1::<i32>() {
            return v[indices[0] as usize] as i64;
        }
        if let Ok(v) = flat.to_vec1::<f32>() {
            return v[indices[0] as usize] as i64;
        }
        0
    }
    pub fn f64_value(&self, indices: &[i64]) -> f64 {
        let flat = self
            .inner
            .inner
            .flatten_all()
            .unwrap()
            .to_device(&candle_core::Device::Cpu)
            .unwrap();
        if let Ok(v) = flat.to_vec1::<f32>() {
            return v[indices[0] as usize] as f64;
        }
        if let Ok(v) = flat.to_vec1::<f64>() {
            return v[indices[0] as usize];
        }
        0.0
    }
    pub fn to_vec_f32(&self) -> Vec<f32> {
        crate::backend::candle::ops::to_vec_f32(&self.inner).unwrap()
    }
}

#[cfg(feature = "mlx")]
impl Tensor {
    pub fn from_mlx(a: crate::backend::mlx::array::MlxArray) -> Self {
        Tensor { inner: a }
    }

    pub fn as_mlx(&self) -> &crate::backend::mlx::array::MlxArray {
        &self.inner
    }

    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i64];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_f32(data, &shape))
    }

    pub fn from_slice_i64(data: &[i64]) -> Self {
        let shape = [data.len() as i64];
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::from_i64(data, &shape))
    }

    pub fn zeros(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::zeros(
            &shape_i32,
            dtype.into(),
        ))
    }

    pub fn ones(shape: &[i64], dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::ones(
            &shape_i32,
            dtype.into(),
        ))
    }

    pub fn full(shape: &[i64], val: f64, dtype: DType, _device: Device) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let val_arr = crate::backend::mlx::array::MlxArray::scalar_f32(val as f32);
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::full(
            &shape_i32,
            &val_arr,
            dtype.into(),
        ))
    }

    pub fn arange(start: i64, end: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(
            start as f64,
            end as f64,
            1.0,
            crate::backend::mlx::ffi::mlx_dtype::MLX_INT64,
        ))
    }

    pub fn arange_f(start: f64, end: f64, step: f64, dtype: DType, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::array::MlxArray::arange(
            start,
            end,
            step,
            dtype.into(),
        ))
    }

    pub fn cat(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::concatenate(&refs, dim as i32))
    }

    pub fn stack(tensors: &[Tensor], dim: i64) -> Self {
        let refs: Vec<&crate::backend::mlx::array::MlxArray> =
            tensors.iter().map(|t| &t.inner).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::stack(&refs, dim as i32))
    }

    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::take(
            &weight.inner,
            &indices.inner,
            0,
        ))
    }

    pub fn hann_window(size: i64, _device: Device) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::hann_window(size as i32))
    }

    pub fn size(&self) -> Vec<i64> {
        self.inner.shape().iter().map(|&s| s as i64).collect()
    }
    pub fn size3(&self) -> (i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2])
    }
    pub fn size4(&self) -> (i64, i64, i64, i64) {
        let s = self.size();
        (s[0], s[1], s[2], s[3])
    }
    pub fn dim(&self) -> usize {
        self.inner.dims().len() as usize
    }

    pub fn view(&self, shape: &[i64]) -> Self {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::reshape(&self.inner, &shape_i32))
    }

    pub fn reshape(&self, shape: &[i64]) -> Self {
        self.view(shape)
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Self {
        let ndim = self.inner.dims().len();
        let d = if dim < 0 { ndim as i64 + dim } else { dim } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::slice(
            &self.inner,
            &[start as i32],
            &[(start + len) as i32],
            &[1i32],
        ))
    }

    pub fn unsqueeze(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim + 1
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::expand_dims(&self.inner, &[d]))
    }

    pub fn squeeze_dim(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(&self.inner, &[d]))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let ndim = self.inner.dims().len();
        let d0 = if dim0 < 0 { ndim as i64 + dim0 } else { dim0 } as i32;
        let d1 = if dim1 < 0 { ndim as i64 + dim1 } else { dim1 } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::swapaxes(&self.inner, d0, d1))
    }

    pub fn permute(&self, dims: &[i64]) -> Self {
        let dims_i32: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
        Tensor::from_mlx(crate::backend::mlx::ops::transpose(&self.inner, &dims_i32))
    }

    pub fn expand(&self, size: &[i64], _implicit: bool) -> Self {
        let current = self.inner.shape();
        let shape_i32: Vec<i32> = size
            .iter()
            .enumerate()
            .map(|(i, &s)| if s == -1 { current[i] } else { s as i32 })
            .collect();
        Tensor::from_mlx(crate::backend::mlx::ops::broadcast_to(
            &self.inner,
            &shape_i32,
        ))
    }

    pub fn contiguous(&self) -> Self {
        self.clone()
    }
    pub fn tr(&self) -> Self {
        self.transpose(-2, -1)
    }
    pub fn get(&self, index: i64) -> Self {
        self.select(0, index)
    }

    pub fn select(&self, dim: i64, index: i64) -> Self {
        let idx = crate::backend::mlx::array::MlxArray::from_i32(&[index as i32], &[1]);
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::squeeze(
            &crate::backend::mlx::ops::take(&self.inner, &idx, d),
            &[d],
        ))
    }

    pub fn matmul(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::matmul(&self.inner, &other.inner))
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        let exp_arr = crate::backend::mlx::array::MlxArray::scalar_f32(exp as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::power(&self.inner, &exp_arr))
    }

    pub fn neg(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::negative(&self.inner))
    }

    pub fn clamp_min(&self, min: f64) -> Self {
        let min_arr = crate::backend::mlx::array::MlxArray::scalar_f32(min as f32);
        Tensor::from_mlx(crate::backend::mlx::ops::maximum(&self.inner, &min_arr))
    }

    pub fn maximum(&self, other: &Tensor) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::maximum(&self.inner, &other.inner))
    }

    pub fn abs(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::abs(&self.inner))
    }

    pub fn square(&self) -> Self {
        self.pow_scalar(2.0)
    }
    pub fn sqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sqrt(&self.inner))
    }
    pub fn rsqrt(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::rsqrt(&self.inner))
    }

    pub fn log10(&self) -> Self {
        let ln_x = crate::backend::mlx::ops::log(&self.inner);
        let ln10 = crate::backend::mlx::array::MlxArray::scalar_f32(std::f32::consts::LN_10);
        Tensor::from_mlx(crate::backend::mlx::ops::divide(&ln_x, &ln10))
    }

    pub fn sin(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::sin(&self.inner))
    }
    pub fn cos(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::cos(&self.inner))
    }
    pub fn exp(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::exp(&self.inner))
    }

    pub fn softmax(&self, dim: i64) -> Self {
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::softmax(&self.inner, &[d]))
    }

    pub fn gelu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::gelu(&self.inner))
    }
    pub fn silu(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::silu(&self.inner))
    }

    pub fn mean_dim(&self, dims: &[i64], keepdim: bool) -> Self {
        let dims_i32: Vec<i32> = dims
            .iter()
            .map(|&d| {
                if d < 0 {
                    self.inner.dims().len() as i32 + d as i32
                } else {
                    d as i32
                }
            })
            .collect();
        Tensor::from_mlx(crate::backend::mlx::ops::mean(
            &self.inner,
            &dims_i32,
            keepdim,
        ))
    }

    pub fn max(&self) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::max_all(&self.inner, false))
    }

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Self {
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim
        } else {
            dim
        } as i32;
        Tensor::from_mlx(crate::backend::mlx::ops::argmax(&self.inner, d, keepdim))
    }

    pub fn triu(&self, diagonal: i64) -> Self {
        Tensor::from_mlx(crate::backend::mlx::ops::triu(&self.inner, diagonal as i32))
    }

    pub fn slice_scatter(&self, src: &Tensor, dim: i64, start: i64, end: i64, _step: i64) -> Self {
        let d = if dim < 0 {
            self.inner.dims().len() as i64 + dim
        } else {
            dim
        } as usize;
        let dim_len = self.size()[d];
        let end = end.clamp(0, dim_len);
        let mut parts: Vec<Tensor> = Vec::new();
        if start > 0 {
            parts.push(self.narrow(dim, 0, start));
        }
        parts.push(src.clone());
        if end < dim_len {
            parts.push(self.narrow(dim, end, dim_len - end));
        }
        Tensor::cat(&parts, dim)
    }

    pub fn fill_(&self, val: f64) -> Self {
        let shape: Vec<i64> = self.size();
        Tensor::full(&shape, val, DType::Float32, Device::Gpu(0))
    }

    pub fn layer_norm(
        &self,
        _normalized_shape: &[i64],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Self {
        if let Some(w) = weight {
            Tensor::from_mlx(crate::backend::mlx::ops::fast_layer_norm(
                &self.inner,
                &w.inner,
                bias.map(|b| &b.inner),
                eps as f32,
            ))
        } else {
            let mean = self.mean_dim(&[-1], true);
            let diff = self - &mean;
            let var = (&diff * &diff).mean_dim(&[-1], true);
            let normalized = &diff / &(var + eps).sqrt();
            if let Some(b) = bias {
                normalized + b
            } else {
                normalized
            }
        }
    }

    pub fn conv2d(
        &self,
        _weight: &Tensor,
        _bias: Option<&Tensor>,
        stride: &[i64],
        padding: &[i64],
        _dilation: &[i64],
        _groups: i64,
    ) -> Self {
        self.matmul(_weight)
    }

    pub fn reflection_pad1d(&self, pad: &[i64]) -> Self {
        Tensor::from_mlx(crate::backend::mlx::signal::reflection_pad1d(
            &self.inner,
            pad[0] as i32,
            pad[1] as i32,
        ))
    }

    pub fn stft(
        &self,
        n_fft: i64,
        hop_length: i64,
        _win_length: i64,
        window: &Tensor,
        _normalized: bool,
        _onesided: bool,
    ) -> Self {
        let mag = crate::backend::mlx::signal::stft_magnitude(
            &self.inner,
            n_fft as i32,
            hop_length as i32,
            &window.inner,
        );
        Tensor::from_mlx(crate::backend::mlx::ops::swapaxes(&mag, 0, 1))
    }

    pub fn to_dtype(&self, dtype: DType) -> Self {
        Tensor::from_mlx(self.inner.astype(dtype.into()))
    }
    pub fn to_device(&self, _device: Device) -> Self {
        self.clone()
    }
    pub fn kind(&self) -> DType {
        DType::from(self.inner.dtype())
    }
    pub fn device(&self) -> Device {
        Device::Gpu(0)
    }
    pub fn shallow_clone(&self) -> Self {
        self.clone()
    }

    pub fn int64_value(&self, _indices: &[i64]) -> i64 {
        0
    }
    pub fn f64_value(&self, _indices: &[i64]) -> f64 {
        0.0
    }
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.inner
            .astype(crate::backend::mlx::ffi::mlx_dtype::MLX_FLOAT32)
            .to_vec_f32()
    }
}

// Operator overloads for candle backend
#[cfg(feature = "candle-backend")]
impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        let lhs = &self.inner.inner;
        let lhs_dev = lhs.device().clone();
        let rhs_on_lhs = rhs.inner.inner.to_device(&lhs_dev).unwrap();
        let rhs_aligned = if lhs.dtype() != rhs_on_lhs.dtype() {
            rhs_on_lhs.to_dtype(lhs.dtype()).unwrap()
        } else {
            rhs_on_lhs
        };

        let result = (&self.inner.inner + &rhs_aligned)
            .or_else(|_| self.inner.inner.broadcast_add(&rhs_aligned))
            .or_else(|_| rhs_aligned.broadcast_add(&self.inner.inner))
            .unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self + &rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        &self + rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        &self + &rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Tensor {
        let result = (&self.inner.inner + (rhs as f64)).unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        &self + rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Tensor {
        let result = (&self.inner.inner - &rhs.inner.inner).unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self - &rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        &self - rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        &self - &rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        let lhs = &self.inner.inner;
        let lhs_dev = lhs.device().clone();
        let rhs_on_lhs = rhs.inner.inner.to_device(&lhs_dev).unwrap();
        let rhs_aligned = if lhs.dtype() != rhs_on_lhs.dtype() {
            rhs_on_lhs.to_dtype(lhs.dtype()).unwrap()
        } else {
            rhs_on_lhs
        };
        let result = (&self.inner.inner * &rhs_aligned)
            .or_else(|_| self.inner.inner.broadcast_mul(&rhs_aligned))
            .or_else(|_| rhs_aligned.broadcast_mul(&self.inner.inner))
            .unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self * &rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        &self * rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        let result = (&self.inner.inner * (rhs as f64)).unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        &self * rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Tensor {
        let lhs = &self.inner.inner;
        let lhs_dev = lhs.device().clone();
        let rhs_on_lhs = rhs.inner.inner.to_device(&lhs_dev).unwrap();
        let rhs_aligned = if lhs.dtype() != rhs_on_lhs.dtype() {
            rhs_on_lhs.to_dtype(lhs.dtype()).unwrap()
        } else {
            rhs_on_lhs
        };

        let result = (&self.inner.inner / &rhs_aligned).or_else(|_| {
            let rhs_shape = rhs_aligned.shape().dims().to_vec();
            let rhs_numel: usize = rhs_shape.iter().product();
            if rhs_numel == 1 {
                let scalar = rhs_aligned
                    .flatten_all()?
                    .to_dtype(candle_core::DType::F32)?
                    .to_vec1::<f32>()?[0];
                &self.inner.inner / (scalar as f64)
            } else {
                Err(candle_core::Error::Msg(format!(
                    "div broadcast not supported for lhs={:?}, rhs={:?}",
                    self.inner.inner.shape().dims(),
                    rhs_aligned.shape().dims()
                )))
            }
        }).unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self / &rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        &self / rhs
    }
}
#[cfg(feature = "candle-backend")]
impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {
        let result = (&self.inner.inner / (rhs as f64)).unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor {
        &self / rhs
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let result = self.inner.inner.clone().neg().unwrap();
        Tensor::from_candle(crate::backend::candle::array::CANDLE_ARRAY::new(result))
    }
}

#[cfg(feature = "candle-backend")]
impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        -&self
    }
}
