pub mod cpu;

pub trait Backend {
    type DType: crate::numeric::Numeric;

    fn add(
        a: &crate::tensor::Tensor<Self::DType, Self>,
        b: &crate::tensor::Tensor<Self::DType, Self>,
    ) -> crate::tensor::Tensor<Self::DType, Self>
    where
        Self: Sized;
    fn sub(
        a: &crate::tensor::Tensor<Self::DType, Self>,
        b: &crate::tensor::Tensor<Self::DType, Self>,
    ) -> crate::tensor::Tensor<Self::DType, Self>
    where
        Self: Sized;
    // Add other backend operations here
}
