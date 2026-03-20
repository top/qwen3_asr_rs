use axum::extract::Multipart;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
    pub file_name: String,
    pub file_data: Vec<u8>,
    pub stream: Option<bool>,
}

impl TranscriptionRequest {
    pub async fn from_multipart(mut multipart: Multipart) -> Result<Self, String> {
        let mut file_name = String::new();
        let mut file_data = Vec::new();
        let mut stream = None;
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

        tracing::info!("Parsed request: file_name={}, file_data_size={}, stream={:?}, fields={}",
                       file_name, file_data.len(), stream, field_count);

        Ok(Self {
            file_name,
            file_data,
            stream,
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
