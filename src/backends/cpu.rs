use crate::{backends::Backend, ops, tensor::Tensor, utils::determinant};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct CPUBackend;

impl Backend for CPUBackend {
    type DType = f32;

    fn add(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.data.shape, b.data.shape);

        let data = a
            .data
            .data
            .iter()
            .zip(&b.data.data)
            .map(|(x, y)| x + y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn sub(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.data.shape, b.data.shape);

        let data = a
            .data
            .data
            .iter()
            .zip(&b.data.data)
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn mul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.data.shape, b.data.shape);

        let data = a
            .data
            .data
            .iter()
            .zip(&b.data.data)
            .map(|(x, y)| x * y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn div_scalar(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|x| x / scalar).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn sub_scalar(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|x| x - scalar).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn neg(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|x| -*x).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn fill(a: &Tensor<Self::DType, Self>, p: Self::DType) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|_| p).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn broadcast(a: &Tensor<Self::DType, Self>, shape: &Vec<usize>) -> Tensor<Self::DType, Self> {
        let new_data = vec![a.data.data[0]; shape.iter().product()];
        Tensor::new(
            0,
            new_data,
            Arc::new(ops::NoOp),
            shape.clone(),
            vec![],
            true,
        )
    }

    /* ---------- Matrix ops ---------- */

    fn matmul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.data.shape.len(), 2);
        assert_eq!(b.data.shape.len(), 2);
        assert_eq!(a.data.shape[1], b.data.shape[0]);

        let (m, k) = (a.data.shape[0], a.data.shape[1]);
        let n = b.data.shape[1];

        let mut out = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a.data.data[i * k + kk] * b.data.data[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }

        Tensor::new(
            0,
            out,
            Arc::new(ops::NoOp),
            vec![m, n],
            vec![a.id, b.id],
            true,
        )
    }

    fn t(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let shape = &a.data.shape;
        let mut new_shape = shape.clone();
        new_shape.reverse();

        let mut new_data = vec![0.0; a.data.data.len()];
        let _new_strides = Tensor::<Self::DType, Self>::calculate_strides(&new_shape);

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                new_data[j * shape[0] + i] = a.data.data[i * shape[1] + j];
            }
        }

        Tensor::new(0, new_data, Arc::new(ops::NoOp), new_shape, vec![], true)
    }

    fn add_broadcast(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        // b is (1, D) or (1,1)
        assert_eq!(a.data.shape.len(), 2);
        assert_eq!(b.data.shape.len(), 2);
        assert_eq!(b.data.shape[0], 1);

        let rows = a.data.shape[0];
        let cols = a.data.shape[1];

        let mut out = vec![0.0; a.data.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] =
                    a.data.data[i * cols + j] + b.data.data[j.min(b.data.data.len() - 1)];
            }
        }

        Tensor::new(
            0,
            out,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    /* ---------- Reductions ---------- */

    fn sum_to_shape(
        grad: &Tensor<Self::DType, Self>,
        target_shape: &[usize],
    ) -> Tensor<Self::DType, Self> {
        let rows = grad.data.shape[0];
        let cols = grad.data.shape[1];

        match target_shape {
            // (1, C)
            [1, c] if *c == cols => {
                let mut out = vec![0.0; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j] += grad.data.data[i * cols + j];
                    }
                }

                Tensor::new(0, out, Arc::new(ops::NoOp), vec![1, cols], vec![], true)
            }

            // (1, 1)
            [1, 1] => {
                let sum = grad.data.data.iter().sum();
                Tensor::new(0, vec![sum], Arc::new(ops::NoOp), vec![1, 1], vec![], true)
            }

            _ => panic!("unsupported broadcast backward shape"),
        }
    }

    fn sum(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let s = a.data.data.iter().sum::<Self::DType>();
        Tensor::new(0, vec![s], Arc::new(ops::NoOp), vec![1], vec![], true)
    }

    fn mean(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let s = a.data.data.iter().sum::<Self::DType>() / a.data.data.len() as Self::DType;
        Tensor::new(0, vec![s], Arc::new(ops::NoOp), vec![1], vec![], true)
    }

    /* ---------- Activations ---------- */

    fn relu(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|x| x.max(0.0)).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn gt(a: &Tensor<Self::DType, Self>, scalar: f32) -> Tensor<Self::DType, Self> {
        let data = a
            .data
            .data
            .iter()
            .map(|x| if *x > scalar { 1.0 } else { 0.0 })
            .collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn sigmoid(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a
            .data
            .data
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn softmax(a: &Tensor<Self::DType, Self>, _dim: usize) -> Tensor<Self::DType, Self> {
        let rows = a.data.shape[0];
        let cols = a.data.shape[1];

        let mut out = vec![0.0; a.data.data.len()];

        for i in 0..rows {
            let row = &a.data.data[i * cols..(i + 1) * cols];
            let max = row
                .iter()
                .cloned()
                .fold(Self::DType::NEG_INFINITY, Self::DType::max);

            let mut sum = 0.0;
            for j in 0..cols {
                let e = (row[j] - max).exp();
                out[i * cols + j] = e;
                sum += e;
            }

            for j in 0..cols {
                out[i * cols + j] /= sum;
            }
        }

        Tensor::new(
            0,
            out,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    /* ---------- Math ops ---------- */

    fn log(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let eps = 1e-7;
        let data = a.data.data.iter().map(|x| (x + eps).ln()).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    fn exp(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.data.iter().map(|x| x.exp()).collect();
        Tensor::new(
            0,
            data,
            Arc::new(ops::NoOp),
            a.data.shape.clone(),
            vec![],
            true,
        )
    }

    // TODO: fix this
    fn inv(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        // first find determinant
        let det = determinant(&a.data);
        if det == 0.0 {
            panic!("Matrix is not invertible");
        }
        let inv_det = 1.0 / det;

        let mut b = a.clone();
        b.data.data = b.data.data.iter().map(|x| x * inv_det).collect();
        b
    }

    /* ---------- Losses (CRITICAL) ---------- */

    fn mse(
        pred: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(pred.data.shape, target.data.shape);

        let mut sum = 0.0;
        for i in 0..pred.data.data.len() {
            let d = pred.data.data[i] - target.data.data[i];
            sum += d * d;
        }

        Tensor::new(
            0,
            vec![sum / pred.data.data.len() as Self::DType],
            Arc::new(ops::NoOp),
            vec![1],
            vec![],
            true,
        )
    }

    /// Numerically stable BCE (NO sigmoid inside user code)
    fn bce_with_logits(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(logits.data.shape, target.data.shape);

        let mut sum = 0.0;
        for i in 0..logits.data.data.len() {
            let z = logits.data.data[i];
            let y = target.data.data[i];

            let max = z.max(0.0);
            sum += max - z * y + ((-z.abs()).exp() + 1.0).ln();
        }

        Tensor::new(
            0,
            vec![sum / logits.data.data.len() as Self::DType],
            Arc::new(ops::NoOp),
            vec![1],
            vec![],
            true,
        )
    }

    fn cross_entropy(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>, // class indices or one-hot
    ) -> Tensor<Self::DType, Self> {
        let probs = Self::softmax(logits, 1);
        let mut sum = 0.0;

        for i in 0..probs.data.data.len() {
            if target.data.data[i] > 0.0 {
                sum -= probs.data.data[i].ln();
            }
        }

        Tensor::new(0, vec![sum], Arc::new(ops::NoOp), vec![1], vec![], true)
    }
}
