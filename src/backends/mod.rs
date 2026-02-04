use crate::{numeric::Numeric, tensor::Tensor};

pub mod cpu;
pub mod wgpu;

pub trait Backend: Clone + 'static {
    type DType: Numeric;
    type Tensor;

    /* ---------- IO ---------- */

    fn from_cpu(data: &[Self::DType], shape: &[usize]) -> Self::Tensor;
    fn to_cpu(t: &Self::Tensor) -> Vec<Self::DType>;

    /* ---------- Elementwise ---------- */

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn sub(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn neg(a: &Self::Tensor) -> Self::Tensor;

    fn div_scalar(a: &Self::Tensor, scalar: Self::DType) -> Self::Tensor;
    fn sub_scalar(a: &Self::Tensor, scalar: Self::DType) -> Self::Tensor;
    fn fill(a: &Self::Tensor, value: Self::DType) -> Self::Tensor;

    /* ---------- Shape ops ---------- */

    fn broadcast(a: &Self::Tensor, out_shape: &[usize]) -> Self::Tensor;
    fn t(a: &Self::Tensor, in_shape: &[usize]) -> Self::Tensor;

    /* ---------- Matrix ops ---------- */

    fn matmul(
        a: &Self::Tensor,
        b: &Self::Tensor,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Self::Tensor;

    /* ---------- Reductions ---------- */

    fn sum(a: &Self::Tensor) -> Self::Tensor;
    fn mean(a: &Self::Tensor) -> Self::Tensor;
    fn sum_to_shape(grad: &Self::Tensor, target_shape: &[usize]) -> Self::Tensor;

    /* ---------- Activations ---------- */

    fn relu(a: &Self::Tensor) -> Self::Tensor;
    fn sigmoid(a: &Self::Tensor) -> Self::Tensor;
    fn gt(a: &Self::Tensor, scalar: Self::DType) -> Self::Tensor;
    fn softmax(a: &Self::Tensor, dim: usize, shape: &[usize]) -> Self::Tensor;

    /* ---------- Math ---------- */

    fn log(a: &Self::Tensor) -> Self::Tensor;
    fn exp(a: &Self::Tensor) -> Self::Tensor;
    fn elementwise_inv(a: &Self::Tensor) -> Self::Tensor;
    fn inv(a: &Self::Tensor, shape: &[usize]) -> Self::Tensor;

    /* ---------- Losses ---------- */

    fn mse(pred: &Self::Tensor, target: &Self::Tensor) -> Self::Tensor;
    fn bce_with_logits(logits: &Self::Tensor, target: &Self::Tensor) -> Self::Tensor;
    fn cross_entropy(logits: &Self::Tensor, target: &Self::Tensor) -> Self::Tensor;
}
