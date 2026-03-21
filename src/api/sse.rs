use axum::response::{IntoResponse, Response};
use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream, StreamExt};
use std::pin::Pin;
use std::convert::Infallible;
use crate::concurrency::ConcurrencyLimiter;
use crate::inference::InferenceEngine;

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
    FeedingSegment {
        state: Option<qwen3_asr::StreamingState>,
        current_segment_idx: usize,
        segment_data: Vec<f32>,
        last_text: String,
        accumulated_text: String,
        segments: Vec<Vec<f32>>,
        _permit: std::sync::Arc<tokio::sync::OwnedSemaphorePermit>, // Using Arc so we can clone it into new states
    },
    FinishingSegment {
        state: Option<qwen3_asr::StreamingState>,
        current_segment_idx: usize,
        accumulated_text: String,
        segments: Vec<Vec<f32>>,
        _permit: std::sync::Arc<tokio::sync::OwnedSemaphorePermit>,
    },
    Done,
}

impl SseResponse {
    fn create_stream(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
        let engine = self.engine.clone();
        let limiter = self.limiter.clone();
        let audio_data = self.audio_data;
        let chunk_size = 19200; // 1.2s bite chunks for processing smoothly
        let sample_rate = 16000;
        let max_duration_sec = 20.0;

        
        // Pre-segment to avoid VRAM overload across entire file
        let mut segments = crate::audio::segment_audio(&audio_data, sample_rate, max_duration_sec);
        if segments.is_empty() {
            segments.push(Vec::new()); // Fallback empty
        }
        
        let stream = stream::unfold(
            (engine, limiter, segments, StreamState::Init),
            move |(engine, limiter, segments, state): (std::sync::Arc<InferenceEngine>, std::sync::Arc<crate::concurrency::ConcurrencyLimiter>, Vec<Vec<f32>>, StreamState)| async move {
                let next: Option<(Option<Event>, _)> = match state {
                    StreamState::Init => {
                        match limiter.acquire_owned().await {
                            Ok(permit) => {
                                let opts = qwen3_asr::StreamingOptions::default()
                                    .with_chunk_size_sec(1.2);
                                let asr_state = engine.init_streaming(opts);

                                let next_state = StreamState::FeedingSegment {
                                    state: Some(asr_state),
                                    current_segment_idx: 0,
                                    segment_data: segments[0].clone(),
                                    last_text: String::new(),
                                    accumulated_text: String::new(),
                                    segments: segments.clone(),
                                    _permit: std::sync::Arc::new(permit),
                                };
                                tracing::info!("SSE: Initialized streaming session ({} segments)", segments.len());
                                Some((None, (engine, limiter, segments, next_state)))
                            }
                            Err(_) => {
                                let evt: Option<Event> = Some(Event::default().json_data(serde_json::json!({"error": "Concurrency limit reached"})).unwrap_or_else(|_| Event::default().data("error")));
                                Some((evt, (engine, limiter, segments, StreamState::Done)))
                            }
                        }
                    }
                    StreamState::FeedingSegment { mut state, current_segment_idx, mut segment_data, last_text, accumulated_text, segments: state_segments, _permit } => {
                        if segment_data.is_empty() {
                            Some((None, (engine, limiter, state_segments.clone(), StreamState::FinishingSegment { 
                                state, 
                                current_segment_idx, 
                                accumulated_text,
                                segments: state_segments,
                                _permit 
                            })))
                        } else {
                            // Drain small chunk from segment chunk to feed progressively
                        let take_len = chunk_size.min(segment_data.len());
                        let chunk: Vec<f32> = segment_data.drain(..take_len).collect();
                        
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

                        let mut asr_state = state.take().expect("State missing in Feeding");
                        match engine.feed_audio(&mut asr_state, &chunk) {
                            Ok(Some(result)) => {
                                if result.text != last_text && !result.text.is_empty() {
                                    tracing::debug!("SSE: Segment partial result: {}", result.text);
                                    
                                    let combined = if accumulated_text.is_empty() {
                                        result.text.clone()
                                    } else {
                                        format!("{} {}", accumulated_text, result.text)
                                    };
                                    
                                    let chunk_json = serde_json::json!({
                                        "text": combined,
                                        "is_final": false
                                    });
                                    let evt: Option<Event> = Some(Event::default().json_data(chunk_json).unwrap_or_else(|_| Event::default().data("error")));
                                    tokio::task::yield_now().await;
                                    Some((evt, (engine, limiter, state_segments.clone(), StreamState::FeedingSegment { 
                                        state: Some(asr_state), 
                                        current_segment_idx,
                                        segment_data,
                                        last_text: result.text,
                                        accumulated_text,
                                        segments: state_segments,
                                        _permit: _permit,
                                    })))
                                } else {
                                    tokio::task::yield_now().await;
                                    Some((None, (engine, limiter, state_segments.clone(), StreamState::FeedingSegment { 
                                        state: Some(asr_state), 
                                        current_segment_idx,
                                        segment_data,
                                        last_text,
                                        accumulated_text,
                                        segments: state_segments,
                                        _permit: _permit,
                                    })))
                                }
                            }
                            Ok(None) => {
                                tokio::task::yield_now().await;
                                Some((None, (engine, limiter, state_segments.clone(), StreamState::FeedingSegment { 
                                    state: Some(asr_state), 
                                    current_segment_idx,
                                    segment_data,
                                    last_text,
                                    accumulated_text,
                                    segments: state_segments,
                                    _permit: _permit,
                                })))
                            }
                            Err(e) => {
                                tracing::error!("SSE: Feeding error: {}", e);
                                let evt: Option<Event> = Some(Event::default().json_data(serde_json::json!({"error": e.to_string()})).unwrap_or_else(|_| Event::default().data("error")));
                                Some((evt, (engine, limiter, state_segments, StreamState::Done)))
                            }
                        }
                        } // end else
                    }
                    StreamState::FinishingSegment { mut state, current_segment_idx, mut accumulated_text, segments: state_segments, _permit } => {
                        let mut asr_state = state.take().expect("State missing in Finishing");
                        match engine.finish_streaming(&mut asr_state) {
                            Ok(result) => {
                                tracing::info!("SSE: Finalized segment {} text: {}", current_segment_idx + 1, result.text);
                                
                                if !accumulated_text.is_empty() && !result.text.is_empty() {
                                    accumulated_text.push(' ');
                                }
                                accumulated_text.push_str(&result.text);

                                let is_very_final = current_segment_idx >= state_segments.len() - 1;
                                
                                let final_json = serde_json::json!({
                                    "text": accumulated_text,
                                    "is_final": is_very_final
                                });
                                let evt: Option<Event> = Some(Event::default().json_data(final_json).unwrap_or_else(|_| Event::default().data("error")));
                                
                                if is_very_final {
                                    Some((evt, (engine, limiter, state_segments, StreamState::Done)))
                                } else {
                                    // Move to next segment
                                    let next_idx = current_segment_idx + 1;
                                    let opts = qwen3_asr::StreamingOptions::default().with_chunk_size_sec(1.2);
                                    let new_asr_state = engine.init_streaming(opts);

                                    
                                    let next_state = StreamState::FeedingSegment {
                                        state: Some(new_asr_state),
                                        current_segment_idx: next_idx,
                                        segment_data: state_segments[next_idx].clone(),
                                        last_text: String::new(),
                                        accumulated_text,
                                        segments: state_segments.clone(),
                                        _permit,
                                    };
                                    Some((evt, (engine, limiter, state_segments, next_state)))
                                }
                            }
                            Err(e) => {
                                tracing::error!("SSE: Finishing error: {}", e);
                                let evt: Option<Event> = Some(Event::default().json_data(serde_json::json!({"error": e.to_string()})).unwrap_or_else(|_| Event::default().data("error")));
                                Some((evt, (engine, limiter, state_segments, StreamState::Done)))
                            }
                        }
                    }
                    StreamState::Done => {
                        None
                    }
                };
                next
            },
        ).filter_map(|evt: Option<Event>| async move {
            evt.map(|e| Ok::<Event, Infallible>(e))
        });

        Box::pin(stream)
    }
}
