use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use crate::tensor::{Device, Tensor};

/// Load all tensors from a model directory.
///
/// Supports both single-file (`model.safetensors`) and sharded
/// (`model.safetensors.index.json` + `model-00001-of-N.safetensors`) formats.
pub fn load_model_weights(model_dir: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    let single_path = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    if single_path.exists() {
        tracing::info!("Loading weights from {:?}", single_path);
        load_safetensors(&single_path, device)
    } else if index_path.exists() {
        tracing::info!("Loading sharded weights from {:?}", index_path);
        load_sharded_safetensors(&index_path, device)
    } else {
        anyhow::bail!(
            "No model weights found in {:?} (expected model.safetensors or model.safetensors.index.json)",
            model_dir
        )
    }
}

/// Load sharded safetensors using the index file.
fn load_sharded_safetensors(index_path: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    let index_data = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index: {:?}", index_path))?;
    let index: serde_json::Value = serde_json::from_str(&index_data)
        .with_context(|| "Failed to parse safetensors index")?;

    let weight_map = index["weight_map"]
        .as_object()
        .context("Missing weight_map in index")?;

    // Collect unique shard filenames
    let mut shard_files: Vec<String> = weight_map.values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let model_dir = index_path.parent().unwrap();
    let mut all_weights = HashMap::new();

    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        tracing::info!("Loading shard: {}", shard_file);
        let shard_weights = load_safetensors(&shard_path, device)
            .with_context(|| format!("Failed to load shard: {}", shard_file))?;
        all_weights.extend(shard_weights);
    }

    Ok(all_weights)
}

/// Load all tensors from a single safetensors file.
#[cfg(feature = "tch-backend")]
pub fn load_safetensors(path: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    let tch_device = tch::Device::from(device);

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors: {:?}", path))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .with_context(|| format!("Failed to mmap safetensors: {:?}", path))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .with_context(|| format!("Failed to deserialize safetensors: {:?}", path))?;

    let mut result = HashMap::new();

    for (name, view) in tensors.iter() {
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        let tensor = match view.dtype() {
            safetensors::Dtype::BF16 => {
                Tensor::from_tch(
                    tch::Tensor::from_data_size(view.data(), &shape, tch::Kind::BFloat16)
                        .to_device(tch_device),
                )
            }
            safetensors::Dtype::F16 => {
                Tensor::from_tch(
                    tch::Tensor::from_data_size(view.data(), &shape, tch::Kind::Half)
                        .to_device(tch_device),
                )
            }
            safetensors::Dtype::F32 => {
                Tensor::from_tch(
                    tch::Tensor::from_data_size(view.data(), &shape, tch::Kind::Float)
                        .to_device(tch_device),
                )
            }
            safetensors::Dtype::I64 => {
                Tensor::from_tch(
                    tch::Tensor::from_data_size(view.data(), &shape, tch::Kind::Int64)
                        .to_device(tch_device),
                )
            }
            dt => anyhow::bail!("Unsupported dtype in safetensors: {:?}", dt),
        };
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

/// Load all tensors from a single safetensors file (MLX backend).
#[cfg(feature = "mlx")]
pub fn load_safetensors(path: &Path, _device: Device) -> Result<HashMap<String, Tensor>> {
    let map = crate::backend::mlx::io::load_safetensors(path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(map
        .into_iter()
        .map(|(name, arr)| (name, Tensor::from_mlx(arr)))
        .collect())
}

/// Load all tensors from a single safetensors file (onnx-runtime backend).
#[cfg(feature = "onnx-runtime")]
pub fn load_safetensors(path: &Path, _device: Device) -> Result<HashMap<String, Tensor>> {
    use crate::backend::onnx_backend::{Tensor as OnnxTensor, DType};
    
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors: {:?}", path))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .with_context(|| format!("Failed to mmap safetensors: {:?}", path))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .with_context(|| format!("Failed to deserialize safetensors: {:?}", path))?;

    let mut result = HashMap::new();

    for (name, view) in tensors.iter() {
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        
        // Convert to f32 tensor (ONNX backend uses f32 internally)
        let data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => view.data().chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as f32).collect(),
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 => {
                // Convert from half precision to float32
                view.data().chunks_exact(2)
                    .map(|chunk| {
                        let v = u16::from_le_bytes(chunk.try_into().unwrap());
                        half::f16::from_bits(v).to_f32()
                    }).collect()
            }
            safetensors::Dtype::I64 => view.data().chunks_exact(8)
                .map(|chunk| {
                    let v = u64::from_le_bytes(chunk.try_into().unwrap());
                    i64::from_le_bytes(v.to_ne_bytes()) as f32
                }).collect(),
            dt => anyhow::bail!("Unsupported dtype in safetensors: {:?}", dt),
        };

        result.insert(name.to_string(), Tensor::from_onnx(OnnxTensor {
            data,
            shape,
            dtype: DType::Float32,
        }));
    }

    Ok(result)
}

/// Load all tensors from a single safetensors file (onnx-runtime backend).
#[cfg(feature = "onnx-runtime")]
pub fn load_safetensors(path: &Path, device: Device) -> Result<HashMap<String, Tensor>> {
    use candle_core::safetensors;
    
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors: {:?}", path))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .with_context(|| format!("Failed to mmap safetensors: {:?}", path))?;

    let tensors = candle_core::safetensors::SafeTensors::deserialize(&mmap)
        .with_context(|| format!("Failed to deserialize safetensors: {:?}", path))?;

    let mut result = HashMap::new();

    for (name, view) in tensors.iter() {
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        let tensor = match view.dtype() {
            safetensors::Dtype::BF16 => {
                Tensor::from(candle_core::Tensor::from_raw_buffer(
                    view.data(), 
                    candle_core::DType::BF16,
                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>().as_slice(),
                    &candle_core::Device::Cpu
                ))
            }
            safetensors::Dtype::F16 => {
                Tensor::from(candle_core::Tensor::from_raw_buffer(
                    view.data(), 
                    candle_core::DType::F16,
                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>().as_slice(),
                    &candle_core::Device::Cpu
                ))
            }
            safetensors::Dtype::F32 => {
                Tensor::from(candle_core::Tensor::from_raw_buffer(
                    view.data(), 
                    candle_core::DType::F32,
                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>().as_slice(),
                    &candle_core::Device::Cpu
                ))
            }
            safetensors::Dtype::I64 => {
                Tensor::from(candle_core::Tensor::from_raw_buffer(
                    view.data(), 
                    candle_core::DType::I64,
                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>().as_slice(),
                    &candle_core::Device::Cpu
                ))
            }
            dt => anyhow::bail!("Unsupported dtype in safetensors: {:?}", dt),
        };
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}


/// Get a tensor from the weights map with a given prefix and suffix.
pub fn get_weight(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    name: &str,
) -> Result<Tensor> {
    let key = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };
    weights
        .get(&key)
        .map(|t| t.shallow_clone())
        .with_context(|| format!("Weight not found: {}", key))
}

/// Get an optional tensor (returns None if not found).
pub fn get_weight_opt(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    name: &str,
) -> Option<Tensor> {
    let key = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    };
    weights.get(&key).map(|t| t.shallow_clone())
}
