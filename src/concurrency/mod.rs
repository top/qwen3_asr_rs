use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::api::AppError;

pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
}

impl ConcurrencyLimiter {
    pub fn new(limit: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(limit)),
        }
    }

    pub async fn acquire(&self) -> Result<tokio::sync::SemaphorePermit<'_>, AppError> {
        self.semaphore.acquire().await.map_err(|_| AppError::ConcurrencyLimit)
    }

    pub async fn acquire_owned(&self) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
        self.semaphore.clone().acquire_owned().await.map_err(|_| AppError::ConcurrencyLimit)
    }
}
