use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream, StreamExt};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

use crate::concurrency::ConcurrencyLimiter;
use crate::inference::InferenceEngine;

pub struct SseResponse {
    pub engine: Arc<InferenceEngine>,
    pub limiter: Arc<ConcurrencyLimiter>,
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
        asr_state: qwen3_asr::StreamingState,
        current_segment_idx: usize,
        /// Read position into segments[current_segment_idx].
        segment_offset: usize,
        last_text: String,
        accumulated_text: String,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    FinishingSegment {
        asr_state: qwen3_asr::StreamingState,
        current_segment_idx: usize,
        accumulated_text: String,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Done,
}

impl SseResponse {
    fn create_stream(self) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
        let engine = self.engine;
        let limiter = self.limiter;
        let audio_data = self.audio_data;

        const CHUNK_SIZE: usize = 19200; // 1.2 s at 16 kHz
        const SAMPLE_RATE: u32 = 16000;
        const MAX_DURATION_SEC: f32 = 20.0;

        let mut segments = crate::audio::segment_audio(&audio_data, SAMPLE_RATE, MAX_DURATION_SEC);
        if segments.is_empty() {
            segments.push(Vec::new());
        }

        // `segments` lives only in the unfold outer-state tuple.
        // It is *moved* in and out on each iteration — never cloned.
        let stream = stream::unfold(
            (engine, limiter, segments, StreamState::Init),
            move |(engine, limiter, segments, state)| async move {
                match state {
                    // ── Acquire permit and initialise the first streaming session ──────
                    StreamState::Init => match limiter.acquire_owned().await {
                        Ok(permit) => {
                            let opts =
                                qwen3_asr::StreamingOptions::default().with_chunk_size_sec(1.2);
                            let asr_state = engine.init_streaming(opts);
                            tracing::info!(
                                "SSE: Initialised streaming session ({} segment(s))",
                                segments.len()
                            );
                            let next = StreamState::FeedingSegment {
                                asr_state,
                                current_segment_idx: 0,
                                segment_offset: 0,
                                last_text: String::new(),
                                accumulated_text: String::new(),
                                _permit: permit,
                            };
                            Some((None, (engine, limiter, segments, next)))
                        }
                        Err(_) => {
                            let evt = make_error_event("Concurrency limit reached");
                            Some((Some(evt), (engine, limiter, segments, StreamState::Done)))
                        }
                    },

                    // ── Feed one chunk at a time from the current segment ─────────────
                    StreamState::FeedingSegment {
                        mut asr_state,
                        current_segment_idx,
                        segment_offset,
                        last_text,
                        accumulated_text,
                        _permit,
                    } => {
                        // Extract the next chunk by slicing (O(1) index arithmetic).
                        // The borrow of `segments` ends at the closing `}` of this block,
                        // so `segments` can be freely moved into the return tuple below.
                        let maybe_chunk = {
                            let seg = &segments[current_segment_idx];
                            if segment_offset >= seg.len() {
                                None
                            } else {
                                let take = CHUNK_SIZE.min(seg.len() - segment_offset);
                                Some((
                                    seg[segment_offset..segment_offset + take].to_vec(),
                                    segment_offset + take,
                                ))
                            }
                        };

                        match maybe_chunk {
                            // Segment fully consumed — move to flush phase
                            None => {
                                let next = StreamState::FinishingSegment {
                                    asr_state,
                                    current_segment_idx,
                                    accumulated_text,
                                    _permit,
                                };
                                Some((None, (engine, limiter, segments, next)))
                            }

                            Some((chunk, new_offset)) => {
                                // Yield before blocking on GPU inference so Hyper has a chance
                                // to flush the previous SSE event to the TCP socket.
                                tokio::task::yield_now().await;

                                match engine.feed_audio(&mut asr_state, &chunk) {
                                    Ok(Some(result))
                                        if !result.text.is_empty() && result.text != last_text =>
                                    {
                                        tracing::debug!("SSE: Partial result: {}", result.text);
                                        let combined = if accumulated_text.is_empty() {
                                            result.text.clone()
                                        } else {
                                            format!("{} {}", accumulated_text, result.text)
                                        };
                                        let evt = make_text_event(&combined, false);
                                        let next = StreamState::FeedingSegment {
                                            asr_state,
                                            current_segment_idx,
                                            segment_offset: new_offset,
                                            last_text: result.text,
                                            accumulated_text,
                                            _permit,
                                        };
                                        Some((Some(evt), (engine, limiter, segments, next)))
                                    }

                                    Ok(_) => {
                                        // No new text yet — advance offset silently
                                        let next = StreamState::FeedingSegment {
                                            asr_state,
                                            current_segment_idx,
                                            segment_offset: new_offset,
                                            last_text,
                                            accumulated_text,
                                            _permit,
                                        };
                                        Some((None, (engine, limiter, segments, next)))
                                    }

                                    Err(e) => {
                                        tracing::error!("SSE: Feed error: {}", e);
                                        let evt = make_error_event(&e.to_string());
                                        Some((
                                            Some(evt),
                                            (engine, limiter, segments, StreamState::Done),
                                        ))
                                    }
                                }
                            }
                        }
                    }

                    // ── Finalise a segment and optionally start the next one ──────────
                    StreamState::FinishingSegment {
                        mut asr_state,
                        current_segment_idx,
                        mut accumulated_text,
                        _permit,
                    } => {
                        // Same reason as above: yield before the final blocking inference
                        // so any previously emitted events are flushed first.
                        tokio::task::yield_now().await;

                        match engine.finish_streaming(&mut asr_state) {
                            Ok(result) => {
                                tracing::info!(
                                    "SSE: Finalised segment {}/{}: {}",
                                    current_segment_idx + 1,
                                    segments.len(),
                                    result.text
                                );
                                if !accumulated_text.is_empty() && !result.text.is_empty() {
                                    accumulated_text.push(' ');
                                }
                                accumulated_text.push_str(&result.text);

                                let is_last = current_segment_idx >= segments.len() - 1;
                                let evt = make_text_event(&accumulated_text, is_last);

                                if is_last {
                                    Some((
                                        Some(evt),
                                        (engine, limiter, segments, StreamState::Done),
                                    ))
                                } else {
                                    let next_idx = current_segment_idx + 1;
                                    let opts = qwen3_asr::StreamingOptions::default()
                                        .with_chunk_size_sec(1.2);
                                    let new_asr_state = engine.init_streaming(opts);
                                    let next = StreamState::FeedingSegment {
                                        asr_state: new_asr_state,
                                        current_segment_idx: next_idx,
                                        segment_offset: 0,
                                        last_text: String::new(),
                                        accumulated_text,
                                        _permit,
                                    };
                                    Some((Some(evt), (engine, limiter, segments, next)))
                                }
                            }
                            Err(e) => {
                                tracing::error!("SSE: Finish error: {}", e);
                                let evt = make_error_event(&e.to_string());
                                Some((Some(evt), (engine, limiter, segments, StreamState::Done)))
                            }
                        }
                    }

                    StreamState::Done => None,
                }
            },
        )
        .filter_map(|evt: Option<Event>| async move { evt.map(|e| Ok::<Event, Infallible>(e)) });

        Box::pin(stream)
    }
}

// ── Event helpers ─────────────────────────────────────────────────────────────

fn make_text_event(text: &str, is_final: bool) -> Event {
    Event::default()
        .json_data(serde_json::json!({ "text": text, "is_final": is_final }))
        .unwrap_or_else(|_| Event::default().data("error"))
}

fn make_error_event(message: &str) -> Event {
    Event::default()
        .json_data(serde_json::json!({ "error": message }))
        .unwrap_or_else(|_| Event::default().data("error"))
}
