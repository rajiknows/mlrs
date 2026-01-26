use crate::{backends::Backend, numeric::Numeric, tensor::Tensor};
pub mod add;
pub mod add_broadcast;
pub mod log;
pub mod neg;
pub mod none;
pub mod sub;

pub trait Operation<T: Numeric, B: Backend<DType = T>> {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B>;
    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>>;
}
