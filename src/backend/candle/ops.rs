use crate::backend::candle::array::CANDLE_ARRAY;
use candle_core::{DType, Result as CResult, Tensor};

// Creation operations
pub fn from_f32(data: &[f32], shape: &[i64]) -> CResult<CANDLE_ARRAY> {
    let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    Ok(CANDLE_ARRAY::new(Tensor::from_vec(
        data.to_vec(),
        shape,
        &candle_core::Device::Cpu,
    )?))
}

pub fn from_i64(data: &[i64], shape: &[i64]) -> CResult<CANDLE_ARRAY> {
    let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let i64_data: Vec<i64> = data.to_vec();
    Ok(CANDLE_ARRAY::new(Tensor::from_vec(
        i64_data,
        shape,
        &candle_core::Device::Cpu,
    )?))
}

pub fn zeros(shape: &[i64], dtype: crate::tensor::DType) -> CResult<CANDLE_ARRAY> {
    let candle_dtype = super::ffi::dtype_to_candle(dtype);
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    Ok(CANDLE_ARRAY::new(Tensor::zeros(
        shape_usize,
        candle_dtype,
        &candle_core::Device::Cpu,
    )?))
}

pub fn ones(shape: &[i64], dtype: crate::tensor::DType) -> CResult<CANDLE_ARRAY> {
    let candle_dtype = super::ffi::dtype_to_candle(dtype);
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    Ok(CANDLE_ARRAY::new(Tensor::ones(
        shape_usize,
        candle_dtype,
        &candle_core::Device::Cpu,
    )?))
}

pub fn full(shape: &[i64], val: f32, dtype: crate::tensor::DType) -> CResult<CANDLE_ARRAY> {
    let candle_dtype = super::ffi::dtype_to_candle(dtype);
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    // Tensor::full is full(shape, val, device) or full(val, shape, device)?
    // Candle 0.9 has ones, zeros taking shape. full is fn full<D: WithDType, S: Into<Shape>>(val: D, shape: S, device: &Device) -> Result<Tensor>
    // but the dtype is inferred. Or we can just build ones and multiply it by val.
    let mut ones_tensor = Tensor::ones(shape_usize, candle_dtype, &candle_core::Device::Cpu)?;
    // cast if val is f32
    if candle_dtype != candle_core::DType::F32 {
        ones_tensor = ones_tensor.to_dtype(candle_core::DType::F32)?;
    }
    let res = (ones_tensor * (val as f64))?.to_dtype(candle_dtype)?;
    Ok(CANDLE_ARRAY::new(res))
}

pub fn arange(start: i64, end: i64) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(Tensor::arange(
        start,
        end,
        &candle_core::Device::Cpu,
    )?))
}

pub fn arange_f(
    start: f32,
    end: f32,
    step: f32,
    dtype: crate::tensor::DType,
) -> CResult<CANDLE_ARRAY> {
    let count = ((end - start) / step).ceil() as usize;
    let data: Vec<f32> = (0..count).map(|i| start + i as f32 * step).collect();
    Ok(CANDLE_ARRAY::new(Tensor::from_vec(
        data,
        vec![count],
        &candle_core::Device::Cpu,
    )?))
}

// Concatenation
pub fn concatenate(tensors: &[&CANDLE_ARRAY], dim: usize) -> CResult<CANDLE_ARRAY> {
    let refs: Vec<&Tensor> = tensors.iter().map(|t| &t.inner).collect();
    Ok(CANDLE_ARRAY::new(Tensor::cat(&refs, dim)?))
}

pub fn stack(tensors: &[&CANDLE_ARRAY], dim: usize) -> CResult<CANDLE_ARRAY> {
    let refs: Vec<&Tensor> = tensors.iter().map(|t| &t.inner).collect();
    Ok(CANDLE_ARRAY::new(Tensor::stack(&refs, dim)?))
}

// Shape operations
pub fn reshape(tensor: &CANDLE_ARRAY, shape: &[i64]) -> CResult<CANDLE_ARRAY> {
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    Ok(CANDLE_ARRAY::new(tensor.inner.reshape(shape_usize)?))
}

pub fn squeeze_dim(tensor: &CANDLE_ARRAY, dim: usize) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.squeeze(dim)?))
}

pub fn unsqueeze(tensor: &CANDLE_ARRAY, dim: usize) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.unsqueeze(dim)?))
}

pub fn permute(tensor: &CANDLE_ARRAY, dims: &[usize]) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.permute(dims)?))
}

pub fn transpose(tensor: &CANDLE_ARRAY, dim0: usize, dim1: usize) -> CResult<CANDLE_ARRAY> {
    let ndim = tensor.inner.dims().len();
    let mut d: Vec<usize> = (0..ndim).collect();
    if ndim >= 2 {
        d.swap(dim0, dim1);
        Ok(CANDLE_ARRAY::new(tensor.inner.permute(d.as_slice())?))
    } else {
        Ok(tensor.clone())
    }
}

// Indexing
pub fn get(tensor: &CANDLE_ARRAY, index: usize) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.get(index)?))
}

pub fn narrow(
    tensor: &CANDLE_ARRAY,
    dim: usize,
    start: usize,
    len: usize,
) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.narrow(dim, start, len)?))
}

// Arithmetic using standard operators
impl std::ops::Add for &CANDLE_ARRAY {
    type Output = CResult<CANDLE_ARRAY>;
    fn add(self, rhs: Self) -> Self::Output {
        Ok(CANDLE_ARRAY::new((&self.inner + &rhs.inner)?))
    }
}

impl std::ops::Sub for &CANDLE_ARRAY {
    type Output = CResult<CANDLE_ARRAY>;
    fn sub(self, rhs: Self) -> Self::Output {
        Ok(CANDLE_ARRAY::new((&self.inner - &rhs.inner)?))
    }
}

impl std::ops::Mul for &CANDLE_ARRAY {
    type Output = CResult<CANDLE_ARRAY>;
    fn mul(self, rhs: Self) -> Self::Output {
        Ok(CANDLE_ARRAY::new((&self.inner * &rhs.inner)?))
    }
}

impl std::ops::Div for &CANDLE_ARRAY {
    type Output = CResult<CANDLE_ARRAY>;
    fn div(self, rhs: Self) -> Self::Output {
        Ok(CANDLE_ARRAY::new((&self.inner / &rhs.inner)?))
    }
}

pub fn matmul(lhs: &CANDLE_ARRAY, rhs: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    fn matmul_with_bf16_fallback(lhs: &Tensor, rhs: &Tensor) -> CResult<Tensor> {
        let rhs = rhs.to_device(lhs.device())?;
        match lhs.matmul(&rhs) {
            Ok(out) => Ok(out),
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("only supported for contiguous tensors") {
                    let lhs_c = lhs.contiguous()?;
                    let rhs_c = rhs.contiguous()?;
                    return lhs_c.matmul(&rhs_c);
                }
                if msg.contains("dtype mismatch") {
                    // Prefer casting the (usually smaller) activation tensor to weight dtype.
                    if let Ok(lhs_cast) = lhs.to_dtype(rhs.dtype()) {
                        if let Ok(out) = lhs_cast.matmul(&rhs) {
                            tracing::debug!(
                                "candle matmul dtype aligned by casting lhs ({:?}->{:?}), lhs={:?}, rhs={:?}",
                                lhs.dtype(),
                                rhs.dtype(),
                                lhs.shape().dims(),
                                rhs.shape().dims()
                            );
                            return Ok(out);
                        }
                    }
                    if let Ok(rhs_cast) = rhs.to_dtype(lhs.dtype()) {
                        if let Ok(out) = lhs.matmul(&rhs_cast) {
                            tracing::debug!(
                                "candle matmul dtype aligned by casting rhs ({:?}->{:?}), lhs={:?}, rhs={:?}",
                                rhs.dtype(),
                                lhs.dtype(),
                                lhs.shape().dims(),
                                rhs.shape().dims()
                            );
                            return Ok(out);
                        }
                    }
                    tracing::warn!(
                        "candle matmul dtype mismatch unresolved, fallback F32 (lhs={:?}, rhs={:?}, err={})",
                        lhs.shape().dims(),
                        rhs.shape().dims(),
                        msg
                    );
                    let lhs_f32 = lhs.to_dtype(DType::F32)?;
                    let rhs_f32 = rhs.to_dtype(DType::F32)?;
                    return lhs_f32.matmul(&rhs_f32);
                }
                if msg.contains("unsupported dtype BF16") {
                    tracing::warn!(
                        "candle matmul BF16 unsupported, fallback F32 (lhs={:?}, rhs={:?}, err={})",
                        lhs.shape().dims(),
                        rhs.shape().dims(),
                        msg
                    );
                    let lhs_f32 = lhs.to_dtype(DType::F32)?;
                    let rhs_f32 = rhs.to_dtype(DType::F32)?;
                    lhs_f32.matmul(&rhs_f32)
                } else {
                    Err(err)
                }
            }
        }
    }

    match matmul_with_bf16_fallback(&lhs.inner, &rhs.inner) {
        Ok(out) => Ok(CANDLE_ARRAY::new(out)),
        Err(err) => {
            let lhs_dims = lhs.inner.shape().dims().to_vec();
            let rhs_dims = rhs.inner.shape().dims().to_vec();
            let is_nd_x_2d =
                lhs_dims.len() >= 3 && rhs_dims.len() == 2 && lhs_dims[lhs_dims.len() - 1] == rhs_dims[0];
            if is_nd_x_2d {
                let k = lhs_dims[lhs_dims.len() - 1];
                let out_features = rhs_dims[1];
                let prefix: usize = lhs_dims[..lhs_dims.len() - 1].iter().product();
                let lhs_2d = lhs.inner.reshape((prefix, k))?;
                let out_2d = matmul_with_bf16_fallback(&lhs_2d, &rhs.inner)?;
                let mut out_shape = lhs_dims[..lhs_dims.len() - 1].to_vec();
                out_shape.push(out_features);
                let out = out_2d.reshape(out_shape)?;
                tracing::debug!(
                    "candle matmul NDx2D fallback: lhs={:?}, rhs={:?}, out={:?}",
                    lhs_dims,
                    rhs_dims,
                    out.shape().dims()
                );
                Ok(CANDLE_ARRAY::new(out))
            } else {
                Err(err)
            }
        }
    }
}

pub fn pow_scalar(tensor: &CANDLE_ARRAY, exp: f32) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.powf(exp.into())?))
}

pub fn neg(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.neg()?))
}

pub fn abs(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.abs()?))
}

pub fn square(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    let x = &tensor.inner;
    Ok(CANDLE_ARRAY::new((x * x)?))
}

pub fn sqrt(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.sqrt()?))
}

pub fn rsqrt(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    let s = tensor.inner.sqrt()?;
    Ok(CANDLE_ARRAY::new(s.recip()?))
}

pub fn exp(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.exp()?))
}

pub fn log10(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    let ln = tensor.inner.log()?;
    Ok(CANDLE_ARRAY::new((&ln / std::f64::consts::LN_10)?))
}

pub fn sin(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.sin()?))
}

pub fn cos(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.cos()?))
}

// Activations
pub fn softmax(tensor: &CANDLE_ARRAY, dim: usize) -> CResult<CANDLE_ARRAY> {
    // Numerically stable softmax: exp(x - max(x)).
    let x_max = tensor.inner.max_keepdim(dim)?;
    let x_max_b = x_max.broadcast_as(tensor.inner.shape())?;
    let shifted = (&tensor.inner - &x_max_b)?;
    let exp_tensor = shifted.exp()?;
    let sum = exp_tensor.sum_keepdim(dim)?;
    let sum_b = sum.broadcast_as(exp_tensor.shape())?;
    Ok(CANDLE_ARRAY::new((&exp_tensor / &sum_b)?))
}

pub fn gelu(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    Ok(CANDLE_ARRAY::new(tensor.inner.gelu_erf()?))
}

pub fn silu(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    let x = &tensor.inner;
    // silu = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    let nx = x.neg()?;
    let exp_nx = nx.exp()?;
    let one = Tensor::ones_like(&exp_nx)?;
    let denom = (&one + &exp_nx)?;
    let result = (x / &denom)?;
    Ok(CANDLE_ARRAY::new(result))
}

// Reductions
pub fn mean(tensor: &CANDLE_ARRAY, dims: &[usize], keepdim: bool) -> CResult<CANDLE_ARRAY> {
    if keepdim {
        Ok(CANDLE_ARRAY::new(tensor.inner.mean_keepdim(dims)?))
    } else {
        Ok(CANDLE_ARRAY::new(tensor.inner.mean(dims)?))
    }
}

pub fn max(tensor: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    // Match torch/max_all semantics used by the unified Tensor API.
    let flat = tensor.inner.flatten_all()?;
    let m = flat.max(0)?;
    Ok(CANDLE_ARRAY::new(m))
}

pub fn argmax(tensor: &CANDLE_ARRAY, dim: usize, keepdim: bool) -> CResult<CANDLE_ARRAY> {
    if keepdim {
        Ok(CANDLE_ARRAY::new(tensor.inner.argmax_keepdim(dim)?))
    } else {
        Ok(CANDLE_ARRAY::new(tensor.inner.argmax(dim)?))
    }
}

// Triu - upper triangular mask
pub fn triu(tensor: &CANDLE_ARRAY, diagonal: i32) -> CResult<CANDLE_ARRAY> {
    let dims = tensor.inner.shape().dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::Msg(format!(
            "triu expects 2D tensor, got shape {:?}",
            dims
        )));
    }
    let rows = dims[0];
    let cols = dims[1];

    let mut mask_data = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if (c as i64) >= (r as i64 + diagonal as i64) {
                mask_data[r * cols + c] = 1;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (rows, cols), tensor.inner.device())?;
    let zeros = Tensor::zeros_like(&tensor.inner)?;
    let out = where_cond(mask, &tensor.inner, &zeros)?;
    Ok(CANDLE_ARRAY::new(out))
}

// Data extraction
pub fn to_vec_f32(tensor: &CANDLE_ARRAY) -> CResult<Vec<f32>> {
    tensor.inner.to_vec1::<f32>()
}

pub fn int64_value(_tensor: &CANDLE_ARRAY, _indices: &[usize]) -> i64 {
    0
}
pub fn f64_value(_tensor: &CANDLE_ARRAY, _indices: &[usize]) -> f64 {
    0.0
}

// Type conversion
pub fn to_dtype(tensor: &CANDLE_ARRAY, dtype: crate::tensor::DType) -> CResult<CANDLE_ARRAY> {
    let candle_dtype = super::ffi::dtype_to_candle(dtype);
    Ok(CANDLE_ARRAY::new(tensor.inner.to_dtype(candle_dtype)?))
}

// Layer norm - simplified
pub fn layer_norm(
    input: &CANDLE_ARRAY,
    _normalized_shape: &[usize],
    weight: Option<&CANDLE_ARRAY>,
    bias: Option<&CANDLE_ARRAY>,
    eps: f32,
) -> CResult<CANDLE_ARRAY> {
    let x = &input.inner;
    let ndim = x.dims().len();
    let last_dim = x.dims()[ndim - 1];

    let x_f32 = x.to_dtype(DType::F32)?;
    let sum = x_f32.sum_keepdim(ndim - 1)?;
    let mean_val = (&sum / (last_dim as f64))?;

    // Use broadcast_as for proper broadcasting
    let shape: Vec<usize> = vec![ndim];
    let diff = (&x_f32 - mean_val.broadcast_as(shape.as_slice())?)?;
    let var = ((&diff * &diff)?.sum_keepdim(ndim - 1)? / (last_dim as f64))?;
    let norm = (&diff * &(var + eps as f64)?.sqrt()?.recip()?)?;

    let mut result = norm.to_dtype(x.dtype())?;
    if let Some(w) = weight {
        result = (&result * &w.inner)?;
    }
    if let Some(b) = bias {
        result = (&result + &b.inner)?;
    }

    Ok(CANDLE_ARRAY::new(result))
}

// Embedding - placeholder
pub fn embedding(_weight: &CANDLE_ARRAY, _indices: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    // Placeholder implementation
    Ok(CANDLE_ARRAY::new(Tensor::zeros(
        &[1, 1],
        DType::F32,
        &candle_core::Device::Cpu,
    )?))
}

// Maximum - element-wise max
pub fn maximum(tensor1: &CANDLE_ARRAY, tensor2: &CANDLE_ARRAY) -> CResult<CANDLE_ARRAY> {
    // Cannot easily do elewise max, we can do element-wise maximum natively in candle:
    let result = tensor1.inner.maximum(&tensor2.inner)?;
    Ok(CANDLE_ARRAY::new(result))
}

// Where conditional
pub fn where_cond(mask: Tensor, x: &Tensor, y: &Tensor) -> CResult<Tensor> {
    let mask_f32 = mask.to_dtype(candle_core::DType::F32)?;
    let cond = mask_f32.ge(&Tensor::new(0.5f32, mask_f32.device())?.broadcast_as(mask_f32.shape())?)?;
    cond.where_cond(x, y)
}
