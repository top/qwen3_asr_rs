// Ensure exactly one backend is selected
#[cfg(all(feature = "candle-backend", feature = "mlx"))]
compile_error!("Features 'candle-backend' and 'mlx' are mutually exclusive");

#[cfg(not(any(feature = "candle-backend", feature = "mlx")))]
compile_error!("Either 'candle-backend' or 'mlx' feature must be enabled");

pub mod tensor;
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

#[cfg(feature = "candle-backend")]
pub mod backend {
    pub mod candle;
}

#[cfg(feature = "mlx")]
pub mod backend {
    pub mod mlx;
}
