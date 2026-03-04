use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use qwen3_asr::inference::AsrInference;
use qwen3_asr::tensor::Device;

#[derive(Parser)]
#[command(name = "asr", about = "Qwen3 ASR - Automatic Speech Recognition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file (one-shot, then exit)
    Transcribe {
        /// Path to the Qwen3-ASR model directory
        model_path: String,

        /// Path to the input audio file (any format supported by ffmpeg)
        audio_file: String,

        /// Optional: force language (e.g., chinese, english, japanese)
        #[arg(short, long)]
        language: Option<String>,
    },

    /// Start an OpenAI-compatible HTTP server (model stays in memory)
    Serve {
        /// Path to the Qwen3-ASR model directory
        model_path: String,

        /// Host address to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Directory to backup uploaded audio files (optional)
        #[arg(long)]
        backup_dir: Option<PathBuf>,

        /// Path to SQLite database for request logging (optional)
        #[arg(long)]
        db_path: Option<PathBuf>,

        /// Default language for all requests (can be overridden per-request)
        #[arg(short, long)]
        language: Option<String>,
    },
}

fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

fn select_device() -> Device {
    #[cfg(feature = "tch-backend")]
    {
        if tch::Cuda::is_available() {
            tracing::info!("Using CUDA device");
            Device::Gpu(0)
        } else {
            tracing::info!("Using CPU device");
            Device::Cpu
        }
    }

    #[cfg(feature = "mlx")]
    {
        qwen3_asr::backend::mlx::stream::init_mlx(true);
        tracing::info!("Using MLX Metal GPU");
        Device::Gpu(0)
    }
}

fn load_model(model_path: &str) -> Result<AsrInference> {
    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_path);
    }
    let device = select_device();
    AsrInference::load(model_dir, device).context("Failed to load model")
}

fn main() -> Result<()> {
    init_logging();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe {
            model_path,
            audio_file,
            language,
        } => {
            if !Path::new(&audio_file).exists() {
                anyhow::bail!("Audio file not found: {}", audio_file);
            }

            let model = load_model(&model_path)?;

            tracing::info!("Transcribing: {}", audio_file);
            let result = model
                .transcribe(&audio_file, language.as_deref())
                .context("Transcription failed")?;

            println!("Language: {}", result.language);
            println!("Text: {}", result.text);
        }

        Commands::Serve {
            model_path,
            host,
            port,
            backup_dir,
            db_path,
            language,
        } => {
            let model = load_model(&model_path)?;
            tracing::info!("Model loaded, starting server...");

            let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
            rt.block_on(qwen3_asr::server::run_server(
                model,
                &host,
                port,
                backup_dir,
                db_path,
                language,
            ))?;
        }
    }

    Ok(())
}
