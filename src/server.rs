use actix_multipart::Multipart;
use actix_web::{web, App, HttpResponse, HttpServer};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::db::{TranscriptionDb, TranscriptionRecord};
use crate::inference::AsrInference;

/// Wrapper to allow AsrInference to be shared across threads.
/// Safety: access is serialized by Mutex, so concurrent mutation cannot occur.
pub(crate) struct SharedModel(AsrInference);
unsafe impl Send for SharedModel {}
unsafe impl Sync for SharedModel {}

/// Shared application state for the HTTP server.
pub struct AppState {
    pub(crate) model: Mutex<SharedModel>,
    pub db: Option<Arc<TranscriptionDb>>,
    pub backup_dir: Option<PathBuf>,
    pub default_language: Option<String>,
}

/// OpenAI-compatible transcription response.
#[derive(serde::Serialize)]
struct TranscriptionResponse {
    text: String,
}

/// Error response body.
#[derive(serde::Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(serde::Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
}

fn error_response(status: actix_web::http::StatusCode, msg: impl ToString) -> HttpResponse {
    HttpResponse::build(status).json(ErrorResponse {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "server_error".to_string(),
        },
    })
}

/// POST /v1/audio/transcriptions
///
/// Accepts multipart/form-data with:
///   - `file`: audio file (required)
///   - `language`: language hint (optional)
async fn transcribe_handler(
    state: web::Data<AppState>,
    mut payload: Multipart,
) -> HttpResponse {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut audio_filename: Option<String> = None;
    let mut language: Option<String> = state.default_language.clone();
    let mut stream = false;

    // Parse multipart fields
    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(f) => f,
            Err(e) => {
                return error_response(
                    actix_web::http::StatusCode::BAD_REQUEST,
                    format!("Failed to read multipart field: {}", e),
                );
            }
        };

        let field_name = field
            .content_disposition()
            .get_name()
            .unwrap_or("")
            .to_string();

        match field_name.as_str() {
            "file" => {
                audio_filename = field
                    .content_disposition()
                    .get_filename()
                    .map(|s| s.to_string());

                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    match chunk {
                        Ok(data) => buf.extend_from_slice(&data),
                        Err(e) => {
                            return error_response(
                                actix_web::http::StatusCode::BAD_REQUEST,
                                format!("Failed to read file data: {}", e),
                            );
                        }
                    }
                }
                audio_bytes = Some(buf);
            }
            "language" => {
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    if let Ok(data) = chunk {
                        buf.extend_from_slice(&data);
                    }
                }
                if let Ok(val) = String::from_utf8(buf) {
                    let val = val.trim().to_string();
                    if !val.is_empty() {
                        language = Some(val);
                    }
                }
            }
            "stream" => {
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    if let Ok(data) = chunk {
                        buf.extend_from_slice(&data);
                    }
                }
                if let Ok(val) = String::from_utf8(buf) {
                    let v = val.trim().to_ascii_lowercase();
                    stream = v == "true" || v == "1";
                }
            }
            _ => {
                // Skip unknown fields
                while field.next().await.is_some() {}
            }
        }
    }

    let audio_bytes = match audio_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            return error_response(
                actix_web::http::StatusCode::BAD_REQUEST,
                "Missing or empty 'file' field in multipart form",
            );
        }
    };

    // Write audio to a temp file (FFmpeg needs a file path)
    let mut tmp_file = match tempfile::Builder::new()
        .prefix("asr_upload_")
        .suffix(".audio")
        .tempfile()
    {
        Ok(f) => f,
        Err(e) => {
            return error_response(
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create temp file: {}", e),
            );
        }
    };

    if let Err(e) = tmp_file.write_all(&audio_bytes) {
        return error_response(
            actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to write temp file: {}", e),
        );
    }

    let tmp_path = if stream {
        // Streaming returns immediately and continues inference in a spawned task.
        // Keep the temp file alive beyond this handler scope.
        let (_f, kept_path) = match tmp_file.keep() {
            Ok(v) => v,
            Err(e) => {
                return error_response(
                    actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to persist temp file for streaming: {}", e),
                );
            }
        };
        kept_path.to_string_lossy().to_string()
    } else {
        tmp_file.path().to_string_lossy().to_string()
    };

    // Optional: backup audio
    let backup_path = if let Some(ref backup_dir) = state.backup_dir {
        let request_id = uuid::Uuid::new_v4().to_string();
        let ext = audio_filename
            .as_deref()
            .and_then(|f| Path::new(f).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("audio");
        let dest = backup_dir.join(format!("{}.{}", request_id, ext));

        if let Err(e) = std::fs::write(&dest, &audio_bytes) {
            tracing::warn!("Failed to backup audio to {:?}: {}", dest, e);
            None
        } else {
            tracing::info!("Audio backed up to {:?}", dest);
            Some((request_id, dest.to_string_lossy().to_string()))
        }
    } else {
        None
    };

    let request_id = backup_path
        .as_ref()
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Run transcription
    let start = Instant::now();

    if stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let state_cloned = state.clone();
        let tmp_path_cloned = tmp_path.clone();
        let lang = language.clone();

        actix_web::rt::spawn(async move {
            let tx_for_infer = tx.clone();
            let tmp_path_for_cleanup = tmp_path_cloned.clone();
            let final_result = web::block(move || {
                let model = state_cloned
                    .model
                    .lock()
                    .map_err(|e| anyhow::anyhow!("model lock poisoned: {}", e))?;
                model
                    .0
                    .transcribe_with_stream(&tmp_path_cloned, lang.as_deref(), Some(tx_for_infer))
            })
            .await;

            match final_result {
                Ok(Ok(r)) => {
                    let _ = tx
                        .send(format!("event: done\ndata: {}\n\n", serde_json::json!({ "text": r.text })))
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx
                        .send(format!(
                            "event: error\ndata: {}\n\n",
                            serde_json::json!({ "message": format!("Transcription failed: {}", e) })
                        ))
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(format!(
                            "event: error\ndata: {}\n\n",
                            serde_json::json!({ "message": format!("Internal error: {}", e) })
                        ))
                        .await;
                }
            }
            if let Err(e) = std::fs::remove_file(&tmp_path_for_cleanup) {
                tracing::warn!("Failed to remove streaming temp file {}: {}", tmp_path_for_cleanup, e);
            }
            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
        });

        let body_stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv()
                .await
                .map(|s| (Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(s)), rx))
        });
        return HttpResponse::Ok()
            .insert_header(("Content-Type", "text/event-stream"))
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("Connection", "keep-alive"))
            .streaming(body_stream);
    }

    let result = {
        let state = state.clone();
        let tmp_path = tmp_path.clone();
        let lang = language.clone();
        // Run the blocking inference on a dedicated thread
        web::block(move || {
            let model = state.model.lock().map_err(|e| anyhow::anyhow!("model lock poisoned: {}", e))?;
            model.0.transcribe(&tmp_path, lang.as_deref())
        }).await
    };

    let duration_ms = start.elapsed().as_millis() as i64;

    match result {
        Ok(Ok(transcribe_result)) => {
            // Log to database
            if let Some(ref db) = state.db {
                let record = TranscriptionRecord {
                    id: request_id,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    audio_filename: audio_filename.clone(),
                    audio_backup_path: backup_path.map(|(_, p)| p),
                    language: transcribe_result.language.clone(),
                    transcription: transcribe_result.text.clone(),
                    duration_ms,
                };
                if let Err(e) = db.insert_record(&record) {
                    tracing::warn!("Failed to log transcription record: {}", e);
                }
            }

            tracing::info!(
                "Transcription completed in {}ms: {} chars",
                duration_ms,
                transcribe_result.text.len()
            );

            HttpResponse::Ok().json(TranscriptionResponse {
                text: transcribe_result.text,
            })
        }
        Ok(Err(e)) => {
            tracing::error!("Transcription failed: {}", e);
            error_response(
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Transcription failed: {}", e),
            )
        }
        Err(e) => {
            tracing::error!("Blocking task failed: {}", e);
            error_response(
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal error: {}", e),
            )
        }
    }
}

/// GET /health
async fn health_handler() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok"
    }))
}

/// Start the HTTP server.
///
/// This function blocks until the server is shut down.
pub async fn run_server(
    model: AsrInference,
    host: &str,
    port: u16,
    backup_dir: Option<PathBuf>,
    db_path: Option<PathBuf>,
    default_language: Option<String>,
) -> Result<()> {
    // Ensure backup directory exists
    if let Some(ref dir) = backup_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create backup directory {:?}", dir))?;
        tracing::info!("Audio backup directory: {:?}", dir);
    }

    // Open database
    let db = if let Some(ref path) = db_path {
        Some(Arc::new(TranscriptionDb::open(path)?))
    } else {
        None
    };

    let state = web::Data::new(AppState {
        model: Mutex::new(SharedModel(model)),
        db,
        backup_dir,
        default_language,
    });

    tracing::info!("Starting ASR server on {}:{}", host, port);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/v1/audio/transcriptions", web::post().to(transcribe_handler))
            .route("/health", web::get().to(health_handler))
    })
    .bind((host, port))
    .with_context(|| format!("Failed to bind to {}:{}", host, port))?
    .run()
    .await
    .context("Server error")?;

    Ok(())
}
