use axum::{routing::{post, get}, Router, response::Html};

use crate::api::{transcribe, transcribe_sse, AppState};

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(|| async { Html(r#"
            <h1>qwen3-asr-server</h1>
            <p>API endpoints:</p>
            <ul>
                <li>POST /v1/audio/transcriptions - Transcribe audio (JSON response)</li>
                <li>POST /v1/audio/transcriptions/stream - Transcribe audio (SSE streaming)</li>
            </ul>
            <p>Test with curl:</p>
            <pre>curl -X POST http://localhost:11433/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "stream=false"</pre>
        "#)}))
        .route("/v1/models", get(crate::api::list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/transcriptions/stream", post(transcribe_sse))
        .with_state(state)
}
