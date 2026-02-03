use crate::{numeric::Numeric, tensor::NdimVector};

pub fn determinant<T: Numeric>(mat: &NdimVector<T>) -> T {
    let n = mat.shape[0];
    assert!(mat.shape.len() == 2 && mat.shape[0] == mat.shape[1]);

    let mut a = mat.data.clone();
    let mut det = T::one();

    for i in 0..n {
        let mut pivot = i;
        while pivot < n && a[pivot * n + i].is_zero() {
            pivot += 1;
        }
        if pivot == n {
            return T::zero();
        }
        if pivot != i {
            for j in 0..n {
                a.swap(i * n + j, pivot * n + j);
            }
            det = -det;
        }

        for j in i + 1..n {
            let factor = a[j * n + i] / a[i * n + i];
            for k in i..n {
                a[j * n + k] = a[j * n + k] - factor * a[i * n + k];
            }
        }
    }

    let mut res = T::one();
    for i in 0..n {
        res = res * a[i * n + i];
    }
    det * res
}

pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;

    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }

    strides
}
