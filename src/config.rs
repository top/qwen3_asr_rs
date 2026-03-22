use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub concurrency_limit: usize,
    pub model_path: String,
    pub model_name: String,
}

impl Config {
    pub fn from_env() -> Self {
        let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "models".to_string());
        let model_name = env::var("MODEL_NAME").unwrap_or_else(|_| {
            std::path::Path::new(&model_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&model_path)
                .to_string()
        });

        Self {
            port: env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            concurrency_limit: env::var("CONCURRENCY_LIMIT")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2),
            model_path,
            model_name,
        }
    }
}
