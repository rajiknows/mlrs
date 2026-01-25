use crate::numeric::Numeric;

use super::Backend;

pub struct CPUBackend;

impl Backend for CPUBackend {
    type DType = f32;

    fn add(
        a: &crate::tensor::Tensor<Self::DType, Self>,
        b: &crate::tensor::Tensor<Self::DType, Self>,
    ) -> crate::tensor::Tensor<Self::DType, Self> {
        assert_eq!(a.shape, b.shape, "Tensor shapes must match for addition");

        let mut data = Vec::with_capacity(a.data.len());
        for i in 0..a.data.len() {
            data.push(a.data[i] + b.data[i]);
        }

        crate::tensor::Tensor {
            id: 0, // This will be set by the graph
            data,
            grad: vec![Self::DType::default(); a.data.len()],
            shape: a.shape.clone(),
            stride: a.stride.clone(),
            op: std::sync::Arc::new(crate::ops::none::None),
            parents: Vec::new(),
            req_grad: true,
        }
    }
    fn sub(
        a: &crate::tensor::Tensor<Self::DType, Self>,
        b: &crate::tensor::Tensor<Self::DType, Self>,
    ) -> crate::tensor::Tensor<Self::DType, Self> {
        assert_eq!(a.shape, b.shape, "Tensor shapes must match for subtraction");

        let mut data = Vec::with_capacity(a.data.len());
        for i in 0..a.data.len() {
            data.push(a.data[i] - b.data[i]);
        }

        crate::tensor::Tensor {
            id: 0, // This will be set by the graph
            data,
            grad: vec![Self::DType::default(); a.data.len()],
            shape: a.shape.clone(),
            stride: a.stride.clone(),
            op: std::sync::Arc::new(crate::ops::none::None),
            parents: Vec::new(),
            req_grad: true,
        }
    }
}

