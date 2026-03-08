// Ensure exactly one backend is selected
#[cfg(all(feature = "tch-backend", feature = "mlx"))]
compile_error!("Features 'tch-backend' and 'mlx' are mutually exclusive");

#[cfg(all(feature = "tch-backend", feature = "onnx-runtime"))]
compile_error!("Features 'tch-backend' and 'onnx-runtime' are mutually exclusive");

#[cfg(all(feature = "mlx", feature = "onnx-runtime"))]
compile_error!("Features 'mlx' and 'onnx-runtime' are mutually exclusive");

#[cfg(not(any(feature = "tch-backend", feature = "mlx", feature = "onnx-runtime")))]
compile_error!("One of 'tch-backend', 'mlx', or 'onnx-runtime' features must be enabled");

pub mod tensor;
#[cfg(feature = "mlx")]
pub mod backend;

// ONNX runtime support
#[cfg(feature = "onnx-runtime")]
pub mod onnx_backend;

pub mod audio;
pub mod audio_encoder;
pub mod config;
pub mod db;
pub mod error;
pub mod inference;
pub mod layers;
pub mod mel;
pub mod server;
pub mod text_decoder;
pub mod tokenizer;
pub mod weights;
