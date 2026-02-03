use crate::{
    backends::Backend,
    numeric::Numeric,
    tensor::{NdimVector, Tensor},
    utils::{calculate_strides, determinant},
};

#[derive(Clone, Copy)]
pub struct CPUBackend;

#[derive(Debug, Clone)]
pub struct CpuTensor<T: Numeric> {
    data: NdimVector<T>,
}
impl Backend for CPUBackend {
    type DType = f32;
    type Tensor = CpuTensor<f32>;

    /* ---------- IO ---------- */

    fn from_cpu(data: &[f32], shape: &[usize]) -> Self::Tensor {
        CpuTensor::new(data.to_vec(), shape.to_vec())
    }

    fn to_cpu(t: &Self::Tensor) -> Vec<f32> {
        t.data.data.clone()
    }

    /* ---------- Elementwise ---------- */

    fn add(a: &Self::Tensor, b: &Self::Tensor, _: &[usize]) -> Self::Tensor {
        CpuTensor::new(
            a.data
                .data
                .iter()
                .zip(&b.data.data)
                .map(|(x, y)| x + y)
                .collect(),
            a.data.shape.clone(),
        )
    }

    fn sub(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data
                .data
                .iter()
                .zip(&b.data.data)
                .map(|(x, y)| x - y)
                .collect(),
            a.data.shape.clone(),
        )
    }

    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data
                .data
                .iter()
                .zip(&b.data.data)
                .map(|(x, y)| x * y)
                .collect(),
            a.data.shape.clone(),
        )
    }

    fn neg(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| -*x).collect(),
            a.data.shape.clone(),
        )
    }

    fn fill(a: &Self::Tensor, p: f32) -> Self::Tensor {
        CpuTensor::new(vec![p; a.data.data.len()], a.data.shape.clone())
    }

    /* ---------- Scalars ---------- */

    fn div_scalar(a: &Self::Tensor, s: f32) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| x / s).collect(),
            a.data.shape.clone(),
        )
    }

    fn sub_scalar(a: &Self::Tensor, s: f32) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| x - s).collect(),
            a.data.shape.clone(),
        )
    }

    /* ---------- Broadcast ---------- */

    fn broadcast(a: &Self::Tensor, shape: &Vec<usize>) -> Self::Tensor {
        CpuTensor::new(vec![a.data.data[0]; shape.iter().product()], shape.clone())
    }

    /* ---------- Matrix ops ---------- */

    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        let (m, k) = (a.data.shape[0], a.data.shape[1]);
        let n = b.data.shape[1];

        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    out[i * n + j] += a.data.data[i * k + kk] * b.data.data[kk * n + j];
                }
            }
        }

        CpuTensor::new(out, vec![m, n])
    }

    fn t(a: &Self::Tensor) -> Self::Tensor {
        let (r, c) = (a.data.shape[0], a.data.shape[1]);
        let mut out = vec![0.0; a.data.data.len()];

        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = a.data.data[i * c + j];
            }
        }

        CpuTensor::new(out, vec![c, r])
    }

    /* ---------- Reductions ---------- */

    fn sum(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(vec![a.data.data.iter().sum()], vec![1])
    }

    fn mean(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            vec![a.data.data.iter().sum::<f32>() / a.data.data.len() as f32],
            vec![1],
        )
    }

    fn sum_to_shape(g: &Self::Tensor, shape: &[usize]) -> Self::Tensor {
        match shape {
            [1, c] => {
                let mut out = vec![0.0; *c];
                for i in 0..g.data.shape[0] {
                    for j in 0..*c {
                        out[j] += g.data.data[i * *c + j];
                    }
                }
                CpuTensor::new(out, shape.to_vec())
            }
            [1, 1] => CpuTensor::new(vec![g.data.data.iter().sum()], vec![1]),
            _ => panic!("unsupported sum_to_shape"),
        }
    }

    /* ---------- Activations ---------- */

    fn relu(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| x.max(0.0)).collect(),
            a.data.shape.clone(),
        )
    }

    fn sigmoid(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data
                .data
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect(),
            a.data.shape.clone(),
        )
    }

    fn softmax(a: &Self::Tensor, _: usize) -> Self::Tensor {
        let (r, c) = (a.data.shape[0], a.data.shape[1]);
        let mut out = vec![0.0; r * c];

        for i in 0..r {
            let row = &a.data.data[i * c..(i + 1) * c];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;

            for j in 0..c {
                out[i * c + j] = (row[j] - max).exp();
                sum += out[i * c + j];
            }
            for j in 0..c {
                out[i * c + j] /= sum;
            }
        }

        CpuTensor::new(out, a.data.shape.clone())
    }

    /* ---------- Losses ---------- */

    fn mse(p: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor {
        let mut s = 0.0;
        for i in 0..p.data.data.len() {
            let d = p.data.data[i] - t.data.data[i];
            s += d * d;
        }
        CpuTensor::new(vec![s / p.data.data.len() as f32], vec![1])
    }

    fn bce_with_logits(l: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor {
        let mut s = 0.0;
        for i in 0..l.data.data.len() {
            let z = l.data.data[i];
            let y = t.data.data[i];
            let m = z.max(0.0);
            s += m - z * y + ((-z.abs()).exp() + 1.0).ln();
        }
        CpuTensor::new(vec![s / l.data.data.len() as f32], vec![1])
    }

    fn cross_entropy(l: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor {
        let p = Self::softmax(l, 1);
        let mut s = 0.0;
        for i in 0..p.data.data.len() {
            if t.data.data[i] > 0.0 {
                s -= p.data.data[i].ln();
            }
        }
        CpuTensor::new(vec![s / l.data.shape[0] as f32], vec![1])
    }

    /* ---------- Misc ---------- */

    fn inv(a: &Self::Tensor) -> Self::Tensor {
        let det = determinant(&a.data);
        assert!(det != 0.0);
        CpuTensor::new(
            a.data.data.iter().map(|x| x / det).collect(),
            a.data.shape.clone(),
        )
    }

    fn elementwise_inv(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| 1.0 / x).collect(),
            a.data.shape.clone(),
        )
    }

    fn gt(a: &Self::Tensor, scalar: f32) -> Self::Tensor {
        CpuTensor::new(
            a.data
                .data
                .iter()
                .map(|x| if *x > scalar { 1.0 } else { 0.0 })
                .collect(),
            a.data.shape.clone(),
        )
    }

    fn log(a: &Self::Tensor) -> Self::Tensor {
        let eps = 1e-7;
        CpuTensor::new(
            a.data.data.iter().map(|x| (x + eps).ln()).collect(),
            a.data.shape.clone(),
        )
    }

    fn exp(a: &Self::Tensor) -> Self::Tensor {
        CpuTensor::new(
            a.data.data.iter().map(|x| x.exp()).collect(),
            a.data.shape.clone(),
        )
    }

    fn add_broadcast(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor {
        let rows = a.data.shape[0];
        let cols = a.data.shape[1];

        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] =
                    a.data.data[i * cols + j] + b.data.data[j.min(b.data.data.len() - 1)];
            }
        }

        CpuTensor::new(out, a.data.shape.clone())
    }
}
impl<T> CpuTensor<T>
where
    T: Numeric,
{
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let stride = calculate_strides(&shape);
        Self {
            data: NdimVector {
                data,
                shape,
                stride,
            },
        }
    }
}
