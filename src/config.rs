use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub concurrency_limit: usize,
    pub model_path: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            concurrency_limit: env::var("CONCURRENCY_LIMIT")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2),
            model_path: env::var("MODEL_PATH").unwrap_or_else(|_| "models".to_string()),
        }
    }
}
