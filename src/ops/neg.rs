use crate::{backends::Backend, numeric::Numeric, ops::Operation, tensor::Tensor};

pub struct Neg;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Neg {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        if inputs.len() > 1 {
            // this is not possible so we should return an error here
            panic!("not possible to have two parents of a neg op");
        }
        B::neg(inputs[0])
    }

    fn backward(&self, output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let neg = B::neg(output_grad);
        vec![neg]
    }
}
