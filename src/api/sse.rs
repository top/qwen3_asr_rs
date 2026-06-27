use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream, StreamExt};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

use crate::concurrency::ConcurrencyLimiter;
use crate::inference::{InferenceEngine, InferenceOptions};

pub struct SseResponse {
    pub engine: Arc<InferenceEngine>,
    pub limiter: Arc<ConcurrencyLimiter>,
    pub audio_data: Vec<f32>,
    pub options: InferenceOptions,
}

pub struct ChatSseResponse {
    pub engine: Arc<InferenceEngine>,
    pub limiter: Arc<ConcurrencyLimiter>,
    pub audio_data: Vec<f32>,
    pub options: InferenceOptions,
    pub model_id: String,
}

#[derive(Clone)]
enum OutputFormat {
    Transcription,
    Chat { model_id: String },
}

impl IntoResponse for SseResponse {
    fn into_response(self) -> Response {
        let stream = build_stream(
            self.engine,
            self.limiter,
            self.audio_data,
            self.options,
            OutputFormat::Transcription,
        );
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    }
}

impl IntoResponse for ChatSseResponse {
    fn into_response(self) -> Response {
        let stream = build_stream(
            self.engine,
            self.limiter,
            self.audio_data,
            self.options,
            OutputFormat::Chat {
                model_id: self.model_id,
            },
        );
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
        // Read position into segments[current_segment_idx].
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

fn build_stream(
    engine: Arc<InferenceEngine>,
    limiter: Arc<ConcurrencyLimiter>,
    audio_data: Vec<f32>,
    options: InferenceOptions,
    output_format: OutputFormat,
) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
    const CHUNK_SIZE: usize = 19200; // 1.2 s at 16 kHz
    const SAMPLE_RATE: u32 = 16000;
    const MAX_DURATION_SEC: f32 = 20.0;

    let mut segments = crate::audio::segment_audio(&audio_data, SAMPLE_RATE, MAX_DURATION_SEC);
    if segments.is_empty() {
        segments.push(Vec::new());
    }

    let stream = stream::unfold(
        (
            engine,
            limiter,
            segments,
            options,
            output_format,
            StreamState::Init,
        ),
        move |(engine, limiter, segments, options, output_format, state)| async move {
            match state {
                StreamState::Init => match limiter.acquire_owned().await {
                    Ok(permit) => {
                        let opts = build_streaming_options(0, &options);
                        let asr_state = engine.init_streaming(opts);
                        let next = StreamState::FeedingSegment {
                            asr_state,
                            current_segment_idx: 0,
                            segment_offset: 0,
                            last_text: String::new(),
                            accumulated_text: String::new(),
                            _permit: permit,
                        };
                        Some((
                            None,
                            (engine, limiter, segments, options, output_format, next),
                        ))
                    }
                    Err(_) => {
                        let evt = make_error_event("Concurrency limit reached");
                        Some((
                            Some(evt),
                            (
                                engine,
                                limiter,
                                segments,
                                options,
                                output_format,
                                StreamState::Done,
                            ),
                        ))
                    }
                },

                StreamState::FeedingSegment {
                    mut asr_state,
                    current_segment_idx,
                    segment_offset,
                    last_text,
                    accumulated_text,
                    _permit,
                } => {
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
                        None => {
                            let next = StreamState::FinishingSegment {
                                asr_state,
                                current_segment_idx,
                                accumulated_text,
                                _permit,
                            };
                            Some((
                                None,
                                (engine, limiter, segments, options, output_format, next),
                            ))
                        }
                        Some((chunk, new_offset)) => {
                            tokio::task::yield_now().await;

                            match engine.feed_audio(&mut asr_state, &chunk) {
                                Ok(Some(result))
                                    if !result.text.is_empty() && result.text != last_text =>
                                {
                                    let combined = if accumulated_text.is_empty() {
                                        result.text.clone()
                                    } else {
                                        format!("{} {}", accumulated_text, result.text)
                                    };
                                    let evt = make_text_event(&combined, false, &output_format);
                                    let next = StreamState::FeedingSegment {
                                        asr_state,
                                        current_segment_idx,
                                        segment_offset: new_offset,
                                        last_text: result.text,
                                        accumulated_text,
                                        _permit,
                                    };
                                    Some((
                                        Some(evt),
                                        (engine, limiter, segments, options, output_format, next),
                                    ))
                                }
                                Ok(_) => {
                                    let next = StreamState::FeedingSegment {
                                        asr_state,
                                        current_segment_idx,
                                        segment_offset: new_offset,
                                        last_text,
                                        accumulated_text,
                                        _permit,
                                    };
                                    Some((
                                        None,
                                        (engine, limiter, segments, options, output_format, next),
                                    ))
                                }
                                Err(e) => {
                                    let evt = make_error_event(&e.to_string());
                                    Some((
                                        Some(evt),
                                        (
                                            engine,
                                            limiter,
                                            segments,
                                            options,
                                            output_format,
                                            StreamState::Done,
                                        ),
                                    ))
                                }
                            }
                        }
                    }
                }

                StreamState::FinishingSegment {
                    mut asr_state,
                    current_segment_idx,
                    mut accumulated_text,
                    _permit,
                } => {
                    tokio::task::yield_now().await;

                    match engine.finish_streaming(&mut asr_state) {
                        Ok(result) => {
                            if !accumulated_text.is_empty() && !result.text.is_empty() {
                                accumulated_text.push(' ');
                            }
                            accumulated_text.push_str(&result.text);

                            let is_last = current_segment_idx >= segments.len() - 1;
                            let evt = make_text_event(&accumulated_text, is_last, &output_format);

                            if is_last {
                                Some((
                                    Some(evt),
                                    (
                                        engine,
                                        limiter,
                                        segments,
                                        options,
                                        output_format,
                                        StreamState::Done,
                                    ),
                                ))
                            } else {
                                let next_idx = current_segment_idx + 1;
                                let opts = build_streaming_options(next_idx, &options);
                                let new_asr_state = engine.init_streaming(opts);
                                let next = StreamState::FeedingSegment {
                                    asr_state: new_asr_state,
                                    current_segment_idx: next_idx,
                                    segment_offset: 0,
                                    last_text: String::new(),
                                    accumulated_text,
                                    _permit,
                                };
                                Some((
                                    Some(evt),
                                    (engine, limiter, segments, options, output_format, next),
                                ))
                            }
                        }
                        Err(e) => {
                            let evt = make_error_event(&e.to_string());
                            Some((
                                Some(evt),
                                (
                                    engine,
                                    limiter,
                                    segments,
                                    options,
                                    output_format,
                                    StreamState::Done,
                                ),
                            ))
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

fn build_streaming_options(segment_idx: usize, options: &InferenceOptions) -> qwen3_asr::StreamingOptions {
    let mut opts = qwen3_asr::StreamingOptions::default().with_chunk_size_sec(1.2);
    if let Some(language) = options.language.as_deref() {
        opts = opts.with_language(language.to_string());
    }
    if segment_idx == 0 {
        if let Some(context) = options.context_text.as_deref() {
            let trimmed = context.trim();
            if !trimmed.is_empty() {
                opts = opts.with_initial_text(trimmed.to_string());
            }
        }
    }
    opts
}

fn make_text_event(text: &str, is_final: bool, output_format: &OutputFormat) -> Event {
    match output_format {
        OutputFormat::Transcription => Event::default()
            .json_data(serde_json::json!({ "text": text, "is_final": is_final }))
            .unwrap_or_else(|_| Event::default().data("error")),
        OutputFormat::Chat { model_id } => {
            let finish_reason = if is_final {
                serde_json::Value::String("stop".to_string())
            } else {
                serde_json::Value::Null
            };

            Event::default()
                .event("chunk")
                .json_data(serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
                    "object": "chat.completion.chunk",
                    "created": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": finish_reason
                    }]
                }))
                .unwrap_or_else(|_| Event::default().data("error"))
        }
    }
}

fn make_error_event(message: &str) -> Event {
    Event::default()
        .event("error")
        .json_data(serde_json::json!({ "error": message }))
        .unwrap_or_else(|_| Event::default().data("error"))
}
