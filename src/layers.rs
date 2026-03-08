use crate::tensor::{DType, Device, Tensor};
use crate::weights::{get_weight, get_weight_opt};
use anyhow::Result;
use std::collections::HashMap;

// ============================================================================
// LayerNorm (with bias, used in audio encoder)
// ============================================================================

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight(weights, prefix, "bias")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let ndim = x.dim();
        x.layer_norm(
            &[x.size()[ndim - 1]],
            Some(&self.weight),
            Some(&self.bias),
            self.eps as f32,
        )
    }
}

// ============================================================================
// RMSNorm (used in text decoder)
// ============================================================================

pub struct RmsNorm {
    pub weight_f32: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        let weight = get_weight(weights, prefix, "weight")?;
        Ok(Self {
            weight_f32: weight.to_dtype(DType::Float32),
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let dtype = x.kind();
        let x = x.to_dtype(DType::Float32);
        let variance = (&x * &x).mean_dim(&[-1], true);
        let x = &x * (&variance + self.eps as f32).rsqrt();
        (x * &self.weight_f32).to_dtype(dtype)
    }
}

// ============================================================================
// Linear layer
// ============================================================================

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight_opt(weights, prefix, "bias"),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = x
            .to_device(self.weight.device())
            .to_dtype(self.weight.kind());
        let out = x.matmul(&self.weight.tr());
        match &self.bias {
            Some(b) => out + b,
            None => out,
        }
    }
}

// ============================================================================
// Conv2d layer
// ============================================================================

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
}

impl Conv2d {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        stride: [i64; 2],
        padding: [i64; 2],
    ) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, prefix, "weight")?,
            bias: get_weight_opt(weights, prefix, "bias"),
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(
            &self.weight,
            self.bias.as_ref(),
            &self.stride,
            &self.padding,
            &[1, 1], // dilation
            1,       // groups
        )
    }
}

// ============================================================================
// Audio encoder self-attention (bidirectional, with bias)
// ============================================================================

pub struct AudioAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AudioAttention {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: Linear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: Linear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (bsz, seq_len, _) = x.size3();
        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        let q = self
            .q_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nh, hd])
            .permute(&[0, 2, 1, 3]);
        let k = self
            .k_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nh, hd])
            .permute(&[0, 2, 1, 3]);
        let v = self
            .v_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nh, hd])
            .permute(&[0, 2, 1, 3]);

        let scale = (hd as f32).sqrt();
        let mut attn = q.matmul(&k.transpose(-2, -1)) / scale;

        if let Some(m) = mask {
            let kind = attn.kind();
            attn = attn + m.to_dtype(kind);
        }

        let attn = attn.softmax(-1).to_dtype(x.kind());
        let out = attn.matmul(&v); // (bsz, nh, seq_len, hd)
        let out = out.permute(&[0, 2, 1, 3]).reshape(&[bsz, seq_len, nh * hd]);
        self.out_proj.forward(&out)
    }
}

// ============================================================================
// Audio encoder FFN
// ============================================================================

pub struct AudioFfn {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl AudioFfn {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: Linear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x).gelu();
        self.fc2.forward(&x)
    }
}

// ============================================================================
// Audio encoder layer
// ============================================================================

pub struct AudioEncoderLayer {
    pub self_attn_layer_norm: LayerNorm,
    pub self_attn: AudioAttention,
    pub final_layer_norm: LayerNorm,
    pub ffn: AudioFfn,
}

impl AudioEncoderLayer {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: LayerNorm::load(
                weights,
                &format!("{}.self_attn_layer_norm", prefix),
                1e-5,
            )?,
            self_attn: AudioAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_heads,
                d_model,
            )?,
            final_layer_norm: LayerNorm::load(
                weights,
                &format!("{}.final_layer_norm", prefix),
                1e-5,
            )?,
            ffn: AudioFfn::load(weights, prefix)?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Pre-norm + self-attention + residual
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x);
        let x = self.self_attn.forward(&x, mask);
        let x = &x + residual;

        // Pre-norm + FFN + residual
        let residual = x.shallow_clone();
        let h = self.final_layer_norm.forward(&x);
        let h = self.ffn.forward(&h);
        h + residual
    }
}

// ============================================================================
// Text decoder attention (GQA with QK-norm and MRoPE)
// ============================================================================

pub struct TextAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

pub struct KvEntry {
    pub k: Tensor,
    pub v: Tensor,
    pub len: i64,
}

impl TextAttention {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(weights, &format!("{}.v_proj", prefix))?,
            o_proj: Linear::load(weights, &format!("{}.o_proj", prefix))?,
            q_norm: RmsNorm::load(weights, &format!("{}.q_norm", prefix), rms_norm_eps)?,
            k_norm: RmsNorm::load(weights, &format!("{}.k_norm", prefix), rms_norm_eps)?,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Forward pass with KV cache support.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<KvEntry>,
        max_seq_len: i64,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let (bsz, seq_len, _) = x.size3();
        let nqh = self.num_q_heads as i64;
        let nkvh = self.num_kv_heads as i64;
        let hd = self.head_dim as i64;

        // Project Q, K, V
        let q = self
            .q_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nqh, hd])
            .transpose(1, 2);
        let k = self
            .k_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nkvh, hd])
            .transpose(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape(&[bsz, seq_len, nkvh, hd])
            .transpose(1, 2);

        // Apply QK normalization (per-head RMSNorm)
        let q = self.apply_head_norm(&q, &self.q_norm);
        let k = self.apply_head_norm(&k, &self.k_norm);

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin);
        let k = apply_rotary_emb(&k, cos, sin);

        // Update KV cache with preallocated storage to avoid per-step cat reallocations.
        let add_len = seq_len;
        if kv_cache.is_none() {
            let init_headroom = std::env::var("ASR_KV_INIT_HEADROOM")
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(256);
            let cap = (add_len + init_headroom).min(max_seq_len).max(add_len);
            let mut k_store = Tensor::zeros(&[bsz, nkvh, cap, hd], k.kind(), k.device());
            let mut v_store = Tensor::zeros(&[bsz, nkvh, cap, hd], v.kind(), v.device());
            let k_src = k.to_device(k_store.device()).to_dtype(k_store.kind());
            let v_src = v.to_device(v_store.device()).to_dtype(v_store.kind());
            k_store = k_store.slice_scatter(&k_src, 2, 0, add_len, 1);
            v_store = v_store.slice_scatter(&v_src, 2, 0, add_len, 1);
            *kv_cache = Some(KvEntry {
                k: k_store,
                v: v_store,
                len: add_len,
            });
        } else if let Some(entry) = kv_cache.as_mut() {
            let start = entry.len;
            let end = start + add_len;
            let current_cap = entry.k.size()[2];
            if end > current_cap {
                let new_cap = (current_cap * 2).max(end).min(max_seq_len);
                assert!(
                    new_cap >= end,
                    "kv cache capacity exceeded: end={} max_seq_len={}",
                    end,
                    max_seq_len
                );
                let mut new_k =
                    Tensor::zeros(&[bsz, nkvh, new_cap, hd], entry.k.kind(), entry.k.device());
                let mut new_v =
                    Tensor::zeros(&[bsz, nkvh, new_cap, hd], entry.v.kind(), entry.v.device());
                if start > 0 {
                    new_k = new_k.slice_scatter(&entry.k.narrow(2, 0, start), 2, 0, start, 1);
                    new_v = new_v.slice_scatter(&entry.v.narrow(2, 0, start), 2, 0, start, 1);
                }
                entry.k = new_k;
                entry.v = new_v;
            }
            let k_src = k.to_device(entry.k.device()).to_dtype(entry.k.kind());
            let v_src = v.to_device(entry.v.device()).to_dtype(entry.v.kind());
            entry.k = entry.k.slice_scatter(&k_src, 2, start, end, 1);
            entry.v = entry.v.slice_scatter(&v_src, 2, start, end, 1);
            entry.len = end;
        }

        let entry = kv_cache.as_ref().expect("kv cache must be initialized");
        let total_kv_len = entry.len;
        let k = entry.k.narrow(2, 0, total_kv_len);
        let v = entry.v.narrow(2, 0, total_kv_len);

        // Compute attention (GQA): avoid materializing repeated KV heads.
        let n_rep = (self.num_q_heads / self.num_kv_heads) as i64;
        let q_grouped = q.reshape(&[bsz, nkvh, n_rep, seq_len, hd]);
        let k_grouped = k
            .unsqueeze(2)
            .expand(&[bsz, nkvh, n_rep, total_kv_len, hd], false); // [bsz, nkvh, n_rep, kv_len, hd]
        let v_grouped = v
            .unsqueeze(2)
            .expand(&[bsz, nkvh, n_rep, total_kv_len, hd], false); // [bsz, nkvh, n_rep, kv_len, hd]

        let scale = (hd as f32).sqrt();
        let bgrp = bsz * nkvh * n_rep;
        let q3 = q_grouped.contiguous().reshape(&[bgrp, seq_len, hd]);
        let k3 = k_grouped
            .contiguous()
            .reshape(&[bgrp, total_kv_len, hd])
            .transpose(-2, -1)
            .contiguous();
        let mut attn_weights = q3.matmul(&k3) / scale;
        attn_weights = attn_weights
            .reshape(&[bsz, nkvh, n_rep, seq_len, total_kv_len])
            .reshape(&[bsz, nqh, seq_len, total_kv_len]);

        if let Some(m) = mask {
            let kind = attn_weights.kind();
            attn_weights = attn_weights + m.to_dtype(kind);
        }

        let attn_weights = attn_weights.softmax(-1).to_dtype(x.kind());
        let attn3 = attn_weights
            .reshape(&[bsz, nkvh, n_rep, seq_len, total_kv_len])
            .contiguous()
            .reshape(&[bgrp, seq_len, total_kv_len]);
        let v3 = v_grouped.contiguous().reshape(&[bgrp, total_kv_len, hd]);
        let out = attn3
            .matmul(&v3)
            .reshape(&[bsz, nkvh, n_rep, seq_len, hd])
            .reshape(&[bsz, nqh, seq_len, hd]);

        // Reshape and project output
        let out = out.transpose(1, 2).reshape(&[bsz, seq_len, nqh * hd]);
        let out = self.o_proj.forward(&out);

        out
    }

    fn apply_head_norm(&self, x: &Tensor, norm: &RmsNorm) -> Tensor {
        norm.forward(x)
    }
}

/// Apply rotary position embeddings.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let cos = cos.unsqueeze(0).unsqueeze(0).to_dtype(x.kind());
    let sin = sin.unsqueeze(0).unsqueeze(0).to_dtype(x.kind());

    let x_rotated = rotate_half(x);
    x * &cos + x_rotated * &sin
}

/// Rotate half: split x into two halves and swap with negation.
fn rotate_half(x: &Tensor) -> Tensor {
    let half = x.size().last().unwrap() / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[(-&x2), x1], -1)
}

// ============================================================================
// Text decoder MLP (SwiGLU)
// ============================================================================

pub struct TextMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl TextMlp {
    pub fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(weights, &format!("{}.gate_proj", prefix))?,
            up_proj: Linear::load(weights, &format!("{}.up_proj", prefix))?,
            down_proj: Linear::load(weights, &format!("{}.down_proj", prefix))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&(gate * up))
    }
}

// ============================================================================
// Text decoder layer
// ============================================================================

pub struct TextDecoderLayer {
    pub input_layernorm: RmsNorm,
    pub self_attn: TextAttention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: TextMlp,
}

impl TextDecoderLayer {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights,
                &format!("{}.input_layernorm", prefix),
                rms_norm_eps,
            )?,
            self_attn: TextAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_q_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                rms_norm_eps,
            )?,
            mlp: TextMlp::load(weights, &format!("{}.mlp", prefix))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<KvEntry>,
        max_seq_len: i64,
        mask: Option<&Tensor>,
    ) -> Tensor {
        // Pre-norm + self-attention + residual
        let residual = x;
        let h = self.input_layernorm.forward(x);
        let h = self
            .self_attn
            .forward(&h, cos, sin, kv_cache, max_seq_len, mask);
        let x = &h + residual;

        // Pre-norm + MLP + residual
        let residual = x.shallow_clone();
        let h = self.post_attention_layernorm.forward(&x);
        let h = self.mlp.forward(&h);
        let out = h + residual;

        out
    }
}

// ============================================================================
// MRoPE (Multimodal Rotary Position Embedding) utilities
// ============================================================================

/// Compute MRoPE cos/sin tensors for the given 3D position IDs.
pub fn compute_mrope_cos_sin(
    position_ids: &[Vec<i64>; 3],
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
    interleaved: bool,
    device: Device,
) -> (Tensor, Tensor) {
    let half_dim = head_dim / 2;
    let seq_len = position_ids[0].len();

    // Compute inverse frequencies
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    // Build dim_map: for each freq index (0..half_dim), which MRoPE dim to use (0,1,2)
    let dim_map = if interleaved {
        build_interleaved_dim_map(mrope_section, half_dim)
    } else {
        build_contiguous_dim_map(mrope_section, half_dim)
    };

    // Compute cos/sin for each position
    let mut cos_vals = vec![0.0f32; seq_len * head_dim];
    let mut sin_vals = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        for j in 0..half_dim {
            let dim = dim_map[j];
            let pos = position_ids[dim][t] as f64;
            let angle = pos * inv_freq[j];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            // First half
            cos_vals[t * head_dim + j] = c;
            sin_vals[t * head_dim + j] = s;
            // Second half (identical, standard RoPE doubling)
            cos_vals[t * head_dim + j + half_dim] = c;
            sin_vals[t * head_dim + j + half_dim] = s;
        }
    }

    let cos = Tensor::from_slice_f32(&cos_vals)
        .reshape(&[seq_len as i64, head_dim as i64])
        .to_device(device);
    let sin = Tensor::from_slice_f32(&sin_vals)
        .reshape(&[seq_len as i64, head_dim as i64])
        .to_device(device);

    (cos, sin)
}

fn build_contiguous_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(total);
    for (dim, &size) in sections.iter().enumerate() {
        for _ in 0..size {
            if map.len() >= total {
                break;
            }
            map.push(dim);
        }
    }
    while map.len() < total {
        map.push(sections.len() - 1);
    }
    map
}

fn build_interleaved_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let n_dims = sections.len();
    let mut map = Vec::with_capacity(total);
    let mut counts = vec![0usize; n_dims];

    while map.len() < total {
        let prev_len = map.len();
        for dim in 0..n_dims {
            if map.len() >= total {
                break;
            }
            if counts[dim] < sections[dim] {
                map.push(dim);
                counts[dim] += 1;
            }
        }
        if map.len() == prev_len {
            break;
        }
    }

    map
}
