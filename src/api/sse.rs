use axum::response::{IntoResponse, Response};
use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream, StreamExt};
use std::pin::Pin;
use std::convert::Infallible;
use std::time::Duration;

use crate::concurrency::ConcurrencyLimiter;
use crate::inference::InferenceEngine;
use tokio::time::sleep;

pub struct SseResponse {
    pub engine: std::sync::Arc<InferenceEngine>,
    pub limiter: std::sync::Arc<ConcurrencyLimiter>,
    pub audio_data: Vec<f32>,
}

impl IntoResponse for SseResponse {
    fn into_response(self) -> Response {
        let stream = self.create_stream();
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    }
}

enum StreamState {
    Init,
    Feeding {
        offset: usize,
        state: qwen3_asr::StreamingState,
        last_text: String,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Finishing {
        state: qwen3_asr::StreamingState,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Done,
    Finished,
}

impl SseResponse {
    fn create_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
        let engine = self.engine.clone();
        let limiter = self.limiter.clone();
        let audio_data = self.audio_data;
        let chunk_size = 8000; // 0.5s chunks for responsiveness

        let stream = stream::unfold(
            (engine, limiter, audio_data, StreamState::Init),
            move |(engine, limiter, audio_data, state)| async move {
                match state {
                    StreamState::Init => {
                        match limiter.acquire_owned().await {
                            Ok(permit) => {
                                let opts = qwen3_asr::StreamingOptions::default()
                                    .with_chunk_size_sec(0.5);
                                let asr_state = engine.init_streaming(opts);
                                let next_state = StreamState::Feeding {
                                    offset: 0,
                                    state: asr_state,
                                    last_text: String::new(),
                                    _permit: permit,
                                };
                                tracing::info!("SSE: Initialized streaming session");
                                return Some((None, (engine, limiter, audio_data, next_state)));
                            }
                            Err(_) => {
                                let evt = Some(Event::default().json_data(serde_json::json!({"error": "Concurrency limit reached"})).unwrap_or(Event::default().data("error")));
                                return Some((evt, (engine, limiter, audio_data, StreamState::Done)));
                            }
                        }
                    }
                    StreamState::Feeding { offset, mut state, last_text, _permit } => {
                        if offset >= audio_data.len() {
                            return Some((None, (engine, limiter, audio_data, StreamState::Finishing { state, _permit })));
                        }

                        let end = (offset + chunk_size).min(audio_data.len());
                        let chunk = &audio_data[offset..end];
                        
                        // Small sleep to ensure flushes and simulate streaming
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

                        match engine.feed_audio(&mut state, chunk) {
                            Ok(Some(result)) => {
                                if result.text != last_text && !result.text.is_empty() {
                                    tracing::info!("SSE: Sending partial result: {}", result.text);
                                    let chunk_json = serde_json::json!({
                                        "text": result.text,
                                        "is_final": false
                                    });
                                    let evt = Some(Event::default().json_data(chunk_json).unwrap_or(Event::default().data("error")));
                                    // Yield to allow flush
                                    tokio::task::yield_now().await;
                                    return Some((evt, (engine, limiter, audio_data, StreamState::Feeding { 
                                        offset: end, 
                                        state, 
                                        last_text: result.text,
                                        _permit: _permit,
                                    })));
                                } else {
                                    // Yield occasionally even if no text to avoid blocking too long
                                    tokio::task::yield_now().await;
                                    return Some((None, (engine, limiter, audio_data, StreamState::Feeding { 
                                        offset: end, 
                                        state, 
                                        last_text,
                                        _permit: _permit,
                                    })));
                                }
                            }
                            Ok(None) => {
                                tokio::task::yield_now().await;
                                return Some((None, (engine, limiter, audio_data, StreamState::Feeding { 
                                    offset: end, 
                                    state, 
                                    last_text,
                                    _permit: _permit,
                                })));
                            }
                            Err(e) => {
                                tracing::error!("SSE: Feeding error: {}", e);
                                let evt = Some(Event::default().json_data(serde_json::json!({"error": e.to_string()})).unwrap_or(Event::default().data("error")));
                                return Some((evt, (engine, limiter, audio_data, StreamState::Done)));
                            }
                        }
                    }
                    StreamState::Finishing { mut state, _permit } => {
                        match engine.finish_streaming(&mut state) {
                            Ok(result) => {
                                tracing::info!("SSE: Sending final result: {}", result.text);
                                let final_json = serde_json::json!({
                                    "text": result.text,
                                    "is_final": true
                                });
                                let evt = Some(Event::default().json_data(final_json).unwrap_or(Event::default().data("error")));
                                return Some((evt, (engine, limiter, audio_data, StreamState::Done)));
                            }
                            Err(e) => {
                                tracing::error!("SSE: Finishing error: {}", e);
                                let evt = Some(Event::default().json_data(serde_json::json!({"error": e.to_string()})).unwrap_or(Event::default().data("error")));
                                return Some((evt, (engine, limiter, audio_data, StreamState::Done)));
                            }
                        }
                    }
                    StreamState::Done => {
                        return None;
                    }
                    StreamState::Finished => {
                        return None;
                    }
                }
            },
        ).filter_map(|evt| async move {
            evt.map(Ok)
        });

        Box::pin(stream)
    }
}
