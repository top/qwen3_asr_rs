use candle_core::Tensor;

#[derive(Clone)]
#[allow(non_camel_case_types)]
pub struct CANDLE_ARRAY {
    pub inner: Tensor,
}

impl std::fmt::Debug for CANDLE_ARRAY {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype()
        )
    }
}

impl CANDLE_ARRAY {
    pub fn new(inner: Tensor) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Tensor {
        &self.inner
    }

    pub fn shape(&self) -> Vec<i64> {
        self.inner.dims().iter().map(|&d| d as i64).collect()
    }

    pub fn dtype(&self) -> candle_core::DType {
        self.inner.dtype()
    }
}

impl From<Tensor> for CANDLE_ARRAY {
    fn from(t: Tensor) -> Self {
        Self::new(t)
    }
}
