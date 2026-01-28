use crate::{numeric::Numeric, tensor::NdimVector};

pub fn determinant<T: Numeric>(mat: &NdimVector<T>) -> Vec<T> {
    let n = mat.shape[0];
    assert!(mat.shape.len() == 2 && mat.shape[0] == mat.shape[1]);

    let mut a = mat.data.clone();
    let mut det = T::one();

    // for each row
    for i in 0..n {
        // there is a pivot that is the diagonal element of the row
        let pivot = i * n + i;

        if a[pivot].is_zero() {
            let mut swap = i + 1;
            while swap < n && a[swap * n + i].is_zero() {
                swap += 1;
            }
            if swap == n {
                return vec![T::zero()];
            }
            for j in 0..n {
                a.swap(i * n + j, swap * n + j);
            }
            det = -det;
        }

        let pivot_val = a[pivot];
        det = det * pivot_val;

        for j in i + 1..n {
            let factor = a[j * n + i] / pivot_val;
            for k in i..n {
                a[j * n + k] = a[j * n + k] - factor * a[i * n + k];
            }
        }
    }

    vec![det]
}
