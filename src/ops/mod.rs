use crate::{backends::Backend, numeric::Numeric, tensor::Tensor};

fn clone_with_shape<B: Backend>(t: &B::Tensor, shape: &[usize]) -> B::Tensor {
    B::from_cpu(&B::to_cpu(t), shape)
}

pub trait Operation<T: Numeric, B: Backend<DType = T>> {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor;
    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor>;
}

pub struct NoOp;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for NoOp {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        clone_with_shape::<B>(&inputs[0].inner, &inputs[0].shape)
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], _output_grad: &B::Tensor) -> Vec<B::Tensor> {
        Vec::<B::Tensor>::new()
    }
}

pub struct Add;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Add {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::add(&inputs[0].inner, &inputs[1].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let grad_lhs = clone_with_shape::<B>(output_grad, &inputs[0].shape);
        let grad_rhs = clone_with_shape::<B>(output_grad, &inputs[1].shape);
        vec![grad_lhs, grad_rhs]
    }
}

pub struct AddBroadcast;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for AddBroadcast {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        if inputs[0].shape == inputs[1].shape {
            B::add(&inputs[0].inner, &inputs[1].inner)
        } else {
            let b = B::broadcast(&inputs[1].inner, &inputs[0].shape);
            B::add(&inputs[0].inner, &b)
        }
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let grad_lhs = clone_with_shape::<B>(output_grad, &inputs[0].shape);
        let grad_rhs = B::sum_to_shape(output_grad, &inputs[1].shape);
        vec![grad_lhs, grad_rhs]
    }
}

pub struct Log;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Log {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::log(&inputs[0].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let x = inputs[0];
        let inv_x = B::elementwise_inv(&x.inner); // 1 / x
        let grad = B::mul(output_grad, &inv_x);
        vec![grad]
    }
}

pub struct Neg;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Neg {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        if inputs.len() > 1 {
            panic!("not possible to have two parents of a neg op");
        }
        B::neg(&inputs[0].inner)
    }

    fn backward(&self, _inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let neg = B::neg(output_grad);
        vec![neg]
    }
}

pub struct Sub;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Sub {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::sub(&inputs[0].inner, &inputs[1].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let grad_lhs = clone_with_shape::<B>(output_grad, &inputs[0].shape);
        let grad_rhs = B::neg(output_grad);
        vec![grad_lhs, grad_rhs]
    }
}

pub struct Mul;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Mul {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::mul(&inputs[0].inner, &inputs[1].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let grad_lhs = B::mul(output_grad, &inputs[1].inner);
        let grad_rhs = B::mul(output_grad, &inputs[0].inner);
        vec![grad_lhs, grad_rhs]
    }
}

pub struct Mean;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Mean {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::mean(&inputs[0].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let input_shape = &inputs[0].shape;
        let n = input_shape.iter().product::<usize>() as f32;
        let n = T::from_f32(n).unwrap();
        let grad = B::div_scalar(output_grad, n);
        let grad = B::broadcast(&grad, input_shape);
        vec![grad]
    }
}

pub struct MatMul;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for MatMul {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::matmul(
            &inputs[0].inner,
            &inputs[1].inner,
            &inputs[0].shape,
            &inputs[1].shape,
        )
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let a = inputs[0];
        let b = inputs[1];
        let bt = B::t(&b.inner, &b.shape);
        let at = B::t(&a.inner, &a.shape);

        let out_shape = vec![a.shape[0], b.shape[1]];
        let bt_shape = vec![b.shape[1], b.shape[0]];
        let at_shape = vec![a.shape[1], a.shape[0]];

        let grad_a = B::matmul(output_grad, &bt, &out_shape, &bt_shape);
        let grad_b = B::matmul(&at, output_grad, &at_shape, &out_shape);
        vec![grad_a, grad_b]
    }
}

pub struct Sigmoid;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for Sigmoid {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::sigmoid(&inputs[0].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let s = self.forward(inputs);
        let s_minus_one = B::sub_scalar(&s, T::one());
        let one_minus_s = B::neg(&s_minus_one);
        let s_mul = B::mul(&s, &one_minus_s);
        let grad = B::mul(output_grad, &s_mul);
        vec![grad]
    }
}

pub struct ReLU;

impl<T: Numeric, B: Backend<DType = T>> Operation<T, B> for ReLU {
    fn forward(&self, inputs: &[&Tensor<T, B>]) -> B::Tensor {
        B::relu(&inputs[0].inner)
    }

    fn backward(&self, inputs: &[&Tensor<T, B>], output_grad: &B::Tensor) -> Vec<B::Tensor> {
        let mask = B::gt(&inputs[0].inner, T::zero());
        let grad = B::mul(output_grad, &mask);
        vec![grad]
    }
}
