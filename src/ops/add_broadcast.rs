use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct AddBroadbast;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for AddBroadbast {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        todo!()
    }
    fn backward(&self, output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        todo!()
    }
}
