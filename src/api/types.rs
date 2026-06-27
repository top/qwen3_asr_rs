use axum::extract::Multipart;
use base64::Engine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
    pub file_name: String,
    pub file_data: Vec<u8>,
    pub stream: Option<bool>,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

impl TranscriptionRequest {
    pub async fn from_multipart(mut multipart: Multipart) -> Result<Self, String> {
        let mut file_name = String::new();
        let mut file_data = Vec::new();
        let mut stream = None;
        let mut model: Option<String> = None;
        let mut language: Option<String> = None;
        let mut prompt: Option<String> = None;
        let mut field_count = 0;
        loop {
            match multipart.next_field().await {
                Ok(Some(field)) => {
                    field_count += 1;
                    let name = field.name().unwrap_or("").to_string();
                    let file_name_field = field.file_name().unwrap_or("").to_string();
                    let content_type = field.content_type().map(|m| m.to_string());

                    tracing::info!("Processing field #{}: name={}, file_name={}, content_type={:?}",
                        field_count, name, file_name_field, content_type);

                    if name == "file" {
                        file_name = file_name_field;
                        match field.bytes().await {
                            Ok(b) => {
                                file_data = b.to_vec();
                                tracing::info!("File field: name={}, size={} bytes", file_name, file_data.len());
                            }
                            Err(e) => {
                                tracing::error!("Error reading file bytes for field #{} (name={}): {:?}", field_count, name, e);
                                return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                            }
                        }
                    } else if name == "stream" {
                        match field.text().await {
                            Ok(s) => {
                                stream = Some(s.to_lowercase() == "true");
                                tracing::info!("Stream field: {}", s);
                            }
                            Err(e) => {
                                tracing::error!("Error reading stream field text for field #{}: {:?}", field_count, e);
                                return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                            }
                        }
                    } else if name == "model" {
                        match field.text().await {
                            Ok(s) => {
                                tracing::info!("Model field: {}", s);
                                model = Some(s);
                            }
                            Err(e) => {
                                tracing::error!("Error reading model field text for field #{}: {:?}", field_count, e);
                                return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                            }
                        }
                    } else if name == "language" {
                        match field.text().await {
                            Ok(s) => {
                                let trimmed = s.trim().to_string();
                                language = if trimmed.is_empty() { None } else { Some(trimmed) };
                            }
                            Err(e) => {
                                tracing::error!("Error reading language field text for field #{}: {:?}", field_count, e);
                                return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                            }
                        }
                    } else if name == "prompt" {
                        match field.text().await {
                            Ok(s) => {
                                let trimmed = s.trim().to_string();
                                prompt = if trimmed.is_empty() { None } else { Some(trimmed) };
                            }
                            Err(e) => {
                                tracing::error!("Error reading prompt field text for field #{}: {:?}", field_count, e);
                                return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                            }
                        }
                    } else {
                        // Log unknown fields
                        match field.text().await {
                            Ok(content) => tracing::warn!("Unknown field: name={}, content={}", name, content),
                            Err(e) => tracing::warn!("Unknown field read error: name={}, err={:?}", name, e),
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("multipart.next_field() error: {:?}", e);
                    return Err(format!("Error parsing `multipart/form-data` request: {}", e));
                }
            }
        }

        tracing::info!("Parsed request: file_name={}, file_data_size={}, stream={:?}, model={:?}, fields={}",
                       file_name, file_data.len(), stream, model, field_count);

        let model = model.ok_or_else(|| "Missing `model` field in request".to_string())?;

        Ok(Self {
            file_name,
            file_data,
            stream,
            model,
            language,
            prompt,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

impl From<crate::inference::TranscriptionResult> for TranscriptionResponse {
    fn from(result: crate::inference::TranscriptionResult) -> Self {
        Self {
            text: result.text,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct ParsedChatInput {
    pub model: String,
    pub stream: bool,
    pub language: Option<String>,
    pub context_text: Option<String>,
    pub audio_data: Vec<f32>,
}

impl ChatCompletionsRequest {
    pub fn into_parsed_input(self) -> Result<ParsedChatInput, String> {
        let mut context_chunks = Vec::new();
        let mut language = None;
        let mut audio_data = None;

        for msg in self.messages {
            match msg.role.as_str() {
                "system" => {
                    if let Some(text) = content_as_text(&msg.content) {
                        let t = text.trim();
                        if !t.is_empty() {
                            context_chunks.push(t.to_string());
                        }
                    }
                }
                "user" => {
                    if let Some((text_chunks, lang, audio)) = parse_user_content(&msg.content)? {
                        for t in text_chunks {
                            let trimmed = t.trim();
                            if !trimmed.is_empty() {
                                context_chunks.push(trimmed.to_string());
                            }
                        }
                        if language.is_none() {
                            language = lang;
                        }
                        if audio_data.is_none() {
                            audio_data = Some(audio);
                        }
                    }
                }
                _ => {}
            }
        }

        let audio_data = audio_data.ok_or_else(|| {
            "No user audio found in messages. Expected content block type=audio_url".to_string()
        })?;

        let context_text = if context_chunks.is_empty() {
            None
        } else {
            Some(context_chunks.join("\n"))
        };

        Ok(ParsedChatInput {
            model: self.model,
            stream: self.stream.unwrap_or(false),
            language,
            context_text,
            audio_data,
        })
    }
}

fn parse_user_content(
    content: &serde_json::Value,
) -> Result<Option<(Vec<String>, Option<String>, Vec<f32>)>, String> {
    let blocks = match content {
        serde_json::Value::Array(items) => items,
        _ => return Ok(None),
    };

    let mut text_chunks = Vec::new();
    let mut language = None;
    let mut audio_data = None;

    for block in blocks {
        let Some(obj) = block.as_object() else {
            continue;
        };
        let typ = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if typ == "text" {
            if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                text_chunks.push(text.to_string());
            }
            continue;
        }

        if typ == "language" {
            if let Some(lang) = obj.get("value").and_then(|v| v.as_str()) {
                let trimmed = lang.trim();
                if !trimmed.is_empty() {
                    language = Some(trimmed.to_string());
                }
            }
            continue;
        }

        if typ == "audio_url" {
            let url = obj
                .get("audio_url")
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "audio_url block missing audio_url.url".to_string())?;

            let bytes = load_audio_from_url(url)?;
            let decoded = crate::audio::process_wav(&bytes)
                .map_err(|e| format!("Failed to decode WAV from audio_url: {}", e))?;
            audio_data = Some(decoded);
        }
    }

    match audio_data {
        Some(audio) => Ok(Some((text_chunks, language, audio))),
        None => Ok(None),
    }
}

fn content_as_text(content: &serde_json::Value) -> Option<String> {
    match content {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(items) => {
            let mut chunks = Vec::new();
            for item in items {
                if let Some(obj) = item.as_object() {
                    let typ = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if typ == "text" {
                        if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                            chunks.push(text.to_string());
                        }
                    }
                }
            }
            if chunks.is_empty() {
                None
            } else {
                Some(chunks.join("\n"))
            }
        }
        _ => None,
    }
}

fn load_audio_from_url(url: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_, b64) = rest
            .split_once(',')
            .ok_or_else(|| "Invalid data URL for audio_url".to_string())?;
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("Invalid base64 audio_url: {}", e));
    }

    Err("Only data: URLs are supported for audio_url in this server".to_string())
}

pub fn new_chat_response(model: &str, text: String) -> ChatCompletionsResponse {
    ChatCompletionsResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
    }
}
