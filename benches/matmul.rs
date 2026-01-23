use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::matrix::{Matrix, mat_mul};
use mlrs::types::B8;

fn bench_matmul_784x1(c: &mut Criterion) {
    let a = Matrix::new(60000, 784);
    let b = Matrix::new(784, 1);
    let mut out = Matrix::new(60000, 1);

    c.bench_function("matmul 60000x784 * 784x1", |bencher| {
        bencher.iter(|| {
            mat_mul(
                std::hint::black_box(&mut out),
                std::hint::black_box(&a),
                std::hint::black_box(&b),
                B8(1),
                B8(0),
                B8(0),
            );
        });
    });
}

criterion_group!(benches, bench_matmul_784x1);
criterion_main!(benches);
