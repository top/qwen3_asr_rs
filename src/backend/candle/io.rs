use crate::backend::candle::array::CANDLE_ARRAY;
use candle_core::{Result as CResult, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub fn load_safetensors(path: &Path, device: &candle_core::Device) -> CResult<HashMap<String, CANDLE_ARRAY>> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let mut result = HashMap::new();
    for (name, tensor) in tensors {
        result.insert(name, CANDLE_ARRAY::new(tensor));
    }
    Ok(result)
}

pub fn load_sharded_safetensors(index_path: &Path, device: &candle_core::Device) -> CResult<HashMap<String, CANDLE_ARRAY>> {
    let index_data = std::fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_data)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| candle_core::Error::Msg("Missing weight_map in index".to_string()))?;

    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let model_dir = index_path.parent().unwrap();
    let mut all_weights = HashMap::new();

    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        tracing::info!("Loading shard: {}", shard_file);
        let shard_weights = load_safetensors(&shard_path, device)?;
        all_weights.extend(shard_weights);
    }

    Ok(all_weights)
}

pub fn load_with_device(
    path: &Path,
    device: crate::tensor::Device,
) -> CResult<HashMap<String, CANDLE_ARRAY>> {
    let c_device = crate::backend::candle::ffi::device_to_candle(device);
    load_safetensors(path, &c_device)
}

