use crate::{backends::Backend, numeric::Numeric, tensor::Tensor};
pub mod add;
pub mod none;
pub mod sub;

pub trait Operation<T: Numeric, B: Backend<DType = T>> {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B>;
    fn backward(&self, output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>>;
}
