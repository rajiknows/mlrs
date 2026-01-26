use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct Log;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Log {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::log(inputs[0])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let x = inputs[0];
        let inv_x = B::inv(x); // 1 / x
        let grad = B::mul(output_grad, &inv_x);
        vec![grad]
    }
}
