use crate::{backends::Backend, numeric::Numeric, ops::Operation};

pub struct None;
impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for None {
    fn forward(&self, inputs: &[&crate::tensor::Tensor<T, B>]) -> crate::tensor::Tensor<T, B> {
        todo!()
    }
    fn backward(
        &self,
        output_grad: &crate::tensor::Tensor<T, B>,
    ) -> Vec<crate::tensor::Tensor<T, B>> {
        todo!()
    }
}
