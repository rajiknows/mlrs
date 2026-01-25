use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct Add;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Add {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        todo!()
    }

    fn backward(&self, output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        todo!()
    }
}
