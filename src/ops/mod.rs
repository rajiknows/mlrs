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

pub struct Add;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Add {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::add(inputs[0], inputs[1])
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        vec![output_grad.clone(), output_grad.clone()]
    }
}

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
