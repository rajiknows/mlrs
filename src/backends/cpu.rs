use crate::backends::Backend;
use crate::tensor::Tensor;

use crate::ops;
use std::sync::Arc;

pub struct CPUBackend;

impl Backend for CPUBackend {
    type DType = f32;

    fn add(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.shape, b.shape);

        let data = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x + y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn sub(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.shape, b.shape);

        let data = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn mul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.shape, b.shape);

        let data = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x * y)
            .collect::<Vec<_>>();

        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn neg(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.iter().map(|x| -*x).collect();
        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn matmul(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(a.shape[1], b.shape[0]);

        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];

        let mut out = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a.data[i * k + kk] * b.data[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }

        Tensor::new(0, out, vec![m, n], Arc::new(ops::none::None), vec![], true)
    }

    fn add_broadcast(
        a: &Tensor<Self::DType, Self>,
        b: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        // b is (1, D) or (1,1)
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(b.shape[0], 1);

        let rows = a.shape[0];
        let cols = a.shape[1];

        let mut out = vec![0.0; a.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = a.data[i * cols + j] + b.data[j.min(b.data.len() - 1)];
            }
        }

        Tensor::new(
            0,
            out,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn sum(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let s = a.data.iter().sum::<Self::DType>();
        Tensor::new(0, vec![s], vec![1], Arc::new(ops::none::None), vec![], true)
    }

    fn mean(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let s = a.data.iter().sum::<Self::DType>() / a.data.len() as Self::DType;
        Tensor::new(0, vec![s], vec![1], Arc::new(ops::none::None), vec![], true)
    }

    fn relu(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.iter().map(|x| x.max(0.0)).collect();
        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn sigmoid(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn softmax(a: &Tensor<Self::DType, Self>, _dim: usize) -> Tensor<Self::DType, Self> {
        let rows = a.shape[0];
        let cols = a.shape[1];

        let mut out = vec![0.0; a.data.len()];

        for i in 0..rows {
            let row = &a.data[i * cols..(i + 1) * cols];
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
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn log(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let eps = 1e-7;
        let data = a.data.iter().map(|x| (x + eps).ln()).collect();
        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn exp(a: &Tensor<Self::DType, Self>) -> Tensor<Self::DType, Self> {
        let data = a.data.iter().map(|x| x.exp()).collect();
        Tensor::new(
            0,
            data,
            a.shape.clone(),
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn mse(
        pred: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(pred.shape, target.shape);

        let mut sum = 0.0;
        for i in 0..pred.data.len() {
            let d = pred.data[i] - target.data[i];
            sum += d * d;
        }

        Tensor::new(
            0,
            vec![sum / pred.data.len() as Self::DType],
            vec![1],
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn bce_with_logits(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        assert_eq!(logits.shape, target.shape);

        let mut sum = 0.0;
        for i in 0..logits.data.len() {
            let z = logits.data[i];
            let y = target.data[i];

            let max = z.max(0.0);
            sum += max - z * y + ((-z.abs()).exp() + 1.0).ln();
        }

        Tensor::new(
            0,
            vec![sum / logits.data.len() as Self::DType],
            vec![1],
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }

    fn cross_entropy(
        logits: &Tensor<Self::DType, Self>,
        target: &Tensor<Self::DType, Self>,
    ) -> Tensor<Self::DType, Self> {
        let probs = Self::softmax(logits, 1);
        let mut sum = 0.0;

        for i in 0..probs.data.len() {
            if target.data[i] > 0.0 {
                sum -= probs.data[i].ln();
            }
        }

        Tensor::new(
            0,
            vec![sum],
            vec![1],
            Arc::new(ops::none::None),
            vec![],
            true,
        )
    }
}
