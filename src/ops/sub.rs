use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct Sub;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Sub {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::sub(inputs[0], inputs[1])
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let grad_lhs = output_grad.clone();
        let grad_rhs = B::neg(output_grad);
        vec![grad_lhs, grad_rhs]
    }
}
