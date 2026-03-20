use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Debug, Serialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub param: Option<String>,
    pub code: Option<String>,
}
