use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct Add;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Add {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::add(inputs[0], inputs[1])
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        vec![output_grad.clone(), output_grad.clone()]
    }
}
