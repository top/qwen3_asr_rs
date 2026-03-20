pub mod routes;
pub mod response;
pub mod sse;
pub mod types;

use axum::extract::{State, Multipart};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

use crate::concurrency::ConcurrencyLimiter;
use crate::inference::InferenceEngine;
use crate::audio::AudioProcessor;
use types::{TranscriptionRequest, TranscriptionResponse};
use response::{OpenAIErrorResponse, OpenAIError};
use sse::SseResponse;

#[derive(Clone)]
pub struct AppState {
    pub inference_engine: Arc<InferenceEngine>,
    pub concurrency_limiter: Arc<ConcurrencyLimiter>,
}

pub async fn transcribe(
    State(state): State<AppState>,
    request: Multipart,
) -> Response {
    tracing::info!("Received transcription request");
    
    let result = async {
        let req = TranscriptionRequest::from_multipart(request)
            .await
            .map_err(|e| {
                tracing::error!("Failed to parse multipart: {}", e);
                AppError::InvalidFormat(e)
            })?;
        
        tracing::info!("File name: {}", req.file_name);
        tracing::info!("File data size: {} bytes", req.file_data.len());
        
        validate_wav_format(&req.file_name)?;
        let audio_data = extract_audio_data(&req.file_data)?;
        
        tracing::info!("Extracted {} audio samples", audio_data.len());
        
        if audio_data.is_empty() {
            return Err(AppError::AudioError("No audio samples extracted".to_string()));
        }
        
        if req.stream.unwrap_or(false) {
            return Ok(EitherResponse::Sse(SseResponse {
                engine: state.inference_engine.clone(),
                limiter: state.concurrency_limiter.clone(),
                audio_data,
            }));
        }
        
        let _guard = state.concurrency_limiter.acquire().await
            .map_err(|_| AppError::ConcurrencyLimit)?;
        let result = state.inference_engine.transcribe(&audio_data)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        Ok(EitherResponse::Json(TranscriptionResponse::from(result)))
    }.await;

    match result {
        Ok(EitherResponse::Json(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(EitherResponse::Sse(response)) => response.into_response(),
        Err(e) => {
            tracing::error!("Transcription failed: {}", e);
            e.into_response()
        }
    }
}

enum EitherResponse {
    Json(TranscriptionResponse),
    Sse(SseResponse),
}

pub async fn transcribe_sse(
    State(state): State<AppState>,
    request: Multipart,
) -> Response {
    let result: Result<SseResponse, AppError> = async {
        let req = TranscriptionRequest::from_multipart(request)
            .await
            .map_err(|e| AppError::InvalidFormat(e))?;
        validate_wav_format(&req.file_name)?;
        let audio_data = extract_audio_data(&req.file_data)?;

        Ok(SseResponse {
            engine: state.inference_engine.clone(),
            limiter: state.concurrency_limiter.clone(),
            audio_data,
        })
    }.await;

    match result {
        Ok(response) => response.into_response(),
        Err(e) => e.into_response(),
    }
}

fn validate_wav_format(file_name: &str) -> Result<(), AppError> {
    if !file_name.ends_with(".wav") {
        return Err(AppError::InvalidFormat("Only WAV format supported".to_string()));
    }
    Ok(())
}

fn extract_audio_data(file_data: &[u8]) -> Result<Vec<f32>, AppError> {
    let audio_processor = AudioProcessor::new();
    audio_processor.process_wav(file_data)
        .map_err(|e| AppError::AudioError(e))
}

#[derive(thiserror::Error, Debug)]
pub enum AppError {
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),
    #[error("Audio processing error: {0}")]
    AudioError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Concurrency limit reached")]
    ConcurrencyLimit,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match self {
            AppError::InvalidFormat(_) => StatusCode::BAD_REQUEST,
            AppError::AudioError(_) => StatusCode::BAD_REQUEST,
            AppError::InferenceError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ConcurrencyLimit => StatusCode::TOO_MANY_REQUESTS,
        };

        let error_response = OpenAIErrorResponse {
            error: OpenAIError {
                message: self.to_string(),
                type_: "invalid_request_error".to_string(),
                param: None,
                code: None,
            },
        };

        (status, Json(error_response)).into_response()
    }
}
