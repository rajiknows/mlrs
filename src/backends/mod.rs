use crate::{numeric::Numeric, tensor::Tensor};

pub mod cpu;

pub trait Backend: Clone {
    type DType: Numeric;

    fn add(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn sub(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn mul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn div_scalar(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn sub_scalar(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn neg(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn fill(a: &Tensor<Self::DType, Self>, x: Self::DType) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn broadcast(
        a: &Tensor<Self::DType, Self>,
        shape: &Vec<usize>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /* ---------- Matrix ops ---------- */

    fn matmul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn t(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn add_broadcast(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>, // (1, D) or (1,1)
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /* ---------- Reductions ---------- */

    fn sum_to_shape(
        grad: &Tensor<Self::DType, Self>,
        target_shape: &[usize],
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn sum(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn mean(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /* ---------- Activations ---------- */

    fn relu(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn gt(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn sigmoid(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn softmax(a: &Tensor<Self::DType, Self>, dim: usize) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /* ---------- Math ops ---------- */

    fn log(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn exp(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn inv(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /* ---------- Losses (CRITICAL) ---------- */

    fn mse(
        pred: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    /// Numerically stable BCE (NO sigmoid inside user code)
    fn bce_with_logits(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;

    fn cross_entropy(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>, // class indices or one-hot
    ) -> Tensor<Self::DType, Self>
    where
        Self: Sized;
}