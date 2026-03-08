use candle_core::{DType, Device as CDDevice};

pub fn dtype_to_candle(dt: crate::tensor::DType) -> DType {
    match dt {
        crate::tensor::DType::Float32 => DType::F32,
        crate::tensor::DType::Float16 => DType::F16,
        crate::tensor::DType::BFloat16 => DType::BF16,
        crate::tensor::DType::Int64 => DType::I64,
        crate::tensor::DType::Int32 => DType::I32,
        crate::tensor::DType::Bool => DType::U8,
    }
}

pub fn candle_dtype_to_dt(dt: &DType) -> crate::tensor::DType {
    match dt {
        DType::F32 => crate::tensor::DType::Float32,
        DType::F16 => crate::tensor::DType::Float16,
        DType::BF16 => crate::tensor::DType::BFloat16,
        DType::I64 => crate::tensor::DType::Int64,
        DType::I32 => crate::tensor::DType::Int32,
        _ => crate::tensor::DType::Float32,
    }
}

pub fn device_to_candle(d: crate::tensor::Device) -> CDDevice {
    #[cfg(feature = "cuda")]
    {
        match d {
            crate::tensor::Device::Cpu => CDDevice::Cpu,
            crate::tensor::Device::Gpu(i) => CDDevice::new_cuda(i).unwrap_or(CDDevice::Cpu),
        }
    }
    #[cfg(not(feature = "cuda"))]
    match d {
        crate::tensor::Device::Cpu => CDDevice::Cpu,
        crate::tensor::Device::Gpu(_) => CDDevice::Cpu,
    }
}

pub fn infer_device() -> crate::tensor::Device {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            return crate::tensor::Device::Gpu(0);
        }
    }
    crate::tensor::Device::Cpu
}
