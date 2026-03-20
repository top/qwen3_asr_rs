mod api;
mod audio;
mod config;
mod concurrency;
mod inference;

use std::net::SocketAddr;
use axum::Router;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    let config = config::Config::from_env();
    let app = build_router(&config)?;
    start_server(&config, app).await
}

fn init_logging() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qwen3_asr_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

fn build_router(config: &config::Config) -> anyhow::Result<Router> {
    let concurrency_limiter = concurrency::ConcurrencyLimiter::new(config.concurrency_limit);
    let inference_engine = inference::InferenceEngine::new(config)?;

    let state = api::AppState {
        inference_engine: std::sync::Arc::new(inference_engine),
        concurrency_limiter: std::sync::Arc::new(concurrency_limiter),
    };

    let app = api::routes::create_router(state)
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    tracing::info_span!(
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                    )
                }),
        );

    Ok(app)
}

async fn start_server(config: &config::Config, app: Router) -> anyhow::Result<()> {
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
