use mlrs::{Matrix, matrix::mat_mul, types::B8};

#[test]
fn test_matmul_nn() {
    let mut a = Matrix::new(2, 2);
    a.data = vec![1.0, 2.0, 3.0, 4.0];

    let mut b = Matrix::new(2, 1);
    b.data = vec![5.0, 6.0];

    let mut out = Matrix::new(2, 1);

    mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));

    assert_close(&out.data, &[17.0, 39.0]);
}

#[test]
fn test_matmul_nt() {
    let mut a = Matrix::new(1, 2);
    a.data = vec![1.0, 2.0];

    let mut b = Matrix::new(1, 2);
    b.data = vec![3.0, 4.0];

    let mut out = Matrix::new(1, 1);

    mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(1));

    // [1,2] · [3,4]^T = 11
    assert_close(&out.data, &[11.0]);
}

#[test]
fn test_matmul_tn() {
    let mut a = Matrix::new(1, 2);
    a.data = vec![1.0, 2.0];

    let mut b = Matrix::new(1, 2);
    b.data = vec![3.0, 4.0];

    let mut out = Matrix::new(2, 2);

    mat_mul(&mut out, &a, &b, B8(1), B8(1), B8(0));

    // [1,2]^T · [3,4]
    // = [[3,4],
    //    [6,8]]
    assert_close(&out.data, &[3.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_matmul_tt() {
    let mut a = Matrix::new(2, 1);
    a.data = vec![1.0, 2.0];

    let mut b = Matrix::new(2, 1);
    b.data = vec![3.0, 4.0];

    let mut out = Matrix::new(1, 1);

    mat_mul(&mut out, &a, &b, B8(1), B8(1), B8(1));

    // [1,2]^T · [3,4]^T = 1*3 + 2*4 = 11
    assert_close(&out.data, &[11.0]);
}

#[test]
fn test_outer_product() {
    let mut a = Matrix::new(2, 1);
    a.data = vec![1.0, 2.0];

    let mut b = Matrix::new(2, 1);
    b.data = vec![3.0, 4.0];

    let mut out = Matrix::new(2, 2);
    mat_mul(&mut out, &a, &b, B8(1), B8(1), B8(1));

    assert_close(&out.data, &[3., 4., 6., 8.]);
}

#[test]
fn test_matmul_zero_out() {
    let mut a = Matrix::new(1, 1);
    a.data = vec![2.0];

    let mut b = Matrix::new(1, 1);
    b.data = vec![3.0];

    let mut out = Matrix::new(1, 1);
    out.data[0] = 100.0;

    mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));

    assert_close(&out.data, &[6.0]);
}

#[test]
fn test_matmul_shape_mismatch() {
    let a = Matrix::new(2, 3);
    let b = Matrix::new(4, 2);
    let mut out = Matrix::new(2, 2);

    let ok = mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));
    assert!(!ok.get());
}
fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 1e-5,
            "Mismatch at {}: {} vs {}",
            i,
            a[i],
            b[i]
        );
    }
}
