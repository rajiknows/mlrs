use std::sync::Arc;

use wgpu::Buffer;

use crate::utils::calculate_strides;

use super::{
    backends::Backend,
    numeric::Numeric,
    ops::{self, Operation},
};

pub type TensorId = usize;

#[derive(Debug, Clone, Default)]
pub struct NdimVector<T: Numeric> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

enum Storage<T: Numeric> {
    Cpu(NdimVector<T>),
    Gpu(Buffer),
}

#[derive(Clone)]
pub struct Tensor<T: Numeric, B: Backend> {
    pub id: TensorId,
    pub grad: Option<B::Tensor>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub inner: B::Tensor,
    pub op: Arc<dyn Operation<T, B>>,
    pub parents: Vec<TensorId>,
    pub req_grad: bool,
}

impl<T: Numeric, B: Backend<DType = T>> Tensor<T, B> {
    pub fn new(
        id: TensorId,
        data: Vec<T>,
        op: Arc<dyn Operation<T, B>>,
        shape: Vec<usize>,
        parents: Vec<TensorId>,
        req_grad: bool,
    ) -> Self {
        let stride = calculate_strides(&shape);
        let inner = B::from_cpu(&data, &shape);

        Self {
            id,
            inner,
            grad: None,
            shape,
            stride,
            op,
            parents,
            req_grad,
        }
    }
    pub fn leaf(id: TensorId, data: Vec<T>, shape: Vec<usize>, req_grad: bool) -> Self {
        let stride = calculate_strides(&shape);
        let inner = B::from_cpu(&data, &shape);

        Self {
            id,
            inner,
            grad: None,
            shape,
            stride,
            op: Arc::new(ops::NoOp),
            parents: Vec::new(),
            req_grad,
        }
    }
}
