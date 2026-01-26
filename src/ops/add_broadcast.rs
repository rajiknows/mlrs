use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct AddBroadbast;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for AddBroadbast {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::add_broadcast(inputs[0], inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let grad_lhs = output_grad.clone();
        let grad_rhs = B::sum_to_shape(output_grad, &inputs[1].shape);
        vec![grad_lhs, grad_rhs]
    }
}
