use crate::{backends::Backend, numeric::Numeric, tensor::Tensor};

pub trait Operation<T: Numeric, B: Backend<DType = T>> {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B>;
    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>>;
}

pub struct NoOp;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for NoOp {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        inputs[0].clone()
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        vec![output_grad.clone()]
    }
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

pub struct AddBroadcast;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for AddBroadcast {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::add_broadcast(inputs[0], inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let grad_lhs = output_grad.clone();
        let grad_rhs = B::sum_to_shape(output_grad, &inputs[1].data.shape);
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

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let neg = B::neg(output_grad);
        vec![neg]
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

pub struct Mul;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Mul {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::mul(inputs[0], inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let grad_lhs = B::mul(output_grad, inputs[1]);
        let grad_rhs = B::mul(output_grad, inputs[0]);
        vec![grad_lhs, grad_rhs]
    }
}

pub struct Mean;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Mean {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::mean(inputs[0])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let input_shape = &inputs[0].data.shape;
        let n = input_shape.iter().product::<usize>() as f32;
        let grad = B::div_scalar(output_grad, n);
        let grad = B::broadcast(&grad, input_shape);
        vec![grad]
    }
}

pub struct MatMul;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for MatMul {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::matmul(inputs[0], inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let a = inputs[0];
        let b = inputs[1];
        let grad_a = B::matmul(output_grad, &(*b).t());
        let grad_b = B::matmul(&(*a).t(), output_grad);
        vec![grad_a, grad_b]
    }
}

pub struct Sigmoid;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Sigmoid {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::sigmoid(inputs[0])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let s = self.forward(inputs);
        let s_minus_one = B::sub_scalar(&s, 1.0);
        let s_mul = B::mul(&s, &s_minus_one);
        let grad = B::mul(output_grad, &s_mul);
        vec![grad]
    }
}

pub struct ReLU;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for ReLU {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Tensor<T, B> {
        B::relu(inputs[0])
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &Tensor<T, B>) -> Vec<Tensor<T, B>> {
        let mask = B::gt(inputs[0], 0.0);
        let grad = B::mul(output_grad, &mask);
        vec![grad]
    }
}

