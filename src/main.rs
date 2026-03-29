mod api;
mod audio;
mod concurrency;
mod config;
mod inference;

use axum::Router;
use std::net::SocketAddr;
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
        model_id: config.model_name.clone(),
    };

    let app = api::routes::create_router(state)
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(
            tower_http::trace::TraceLayer::new_for_http().make_span_with(
                |request: &axum::http::Request<_>| {
                    tracing::info_span!(
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                    )
                },
            ),
        );

    Ok(app)
}

async fn start_server(config: &config::Config, app: Router) -> anyhow::Result<()> {
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use tower::Service;

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on http://{}", addr);

    loop {
        let (socket, _remote_addr) = listener.accept().await?;

        // Disable Nagle's algorithm on every accepted connection so each SSE
        // event is sent to the client immediately rather than being batched
        // with subsequent packets (up to ~200 ms delay otherwise).
        if let Err(e) = socket.set_nodelay(true) {
            tracing::warn!("Failed to set TCP_NODELAY: {}", e);
        }

        // Clone the router for this connection — Router is cheap to clone
        // (it holds an Arc internally).
        let app = app.clone();

        tokio::spawn(async move {
            let io = TokioIo::new(socket);

            // hyper delivers Request<Incoming>; axum expects Request<axum::body::Body>.
            // Map the body type before handing the request to the router.
            // Clone app per-request: service_fn requires Fn (not FnMut),
            // but Service::call needs &mut self, so each call gets its own clone.
            let hyper_svc =
                hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    let req = req.map(axum::body::Body::new);
                    let mut app = app.clone();
                    Service::call(&mut app, req)
                });

            if let Err(e) = Builder::new(TokioExecutor::new())
                .serve_connection_with_upgrades(io, hyper_svc)
                .await
            {
                // Client disconnects and similar errors are normal.
                tracing::debug!("Connection closed: {}", e);
            }
        });
    }
}
