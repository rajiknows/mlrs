use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mlrs::matrix::{Matrix, mat_mul};
use mlrs::types::B8;

fn bench_small_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_matmul");

    for size in [8, 16, 32, 64] {
        let a = Matrix::new(size, size);
        let b = Matrix::new(size, size);
        let mut out = Matrix::new(size, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));
            });
        });
    }

    group.finish();
}

fn bench_medium_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_matmul");

    for (m, k, n) in [(256, 256, 256), (512, 512, 512)] {
        let a = Matrix::new(m, k);
        let b = Matrix::new(k, n);
        let mut out = Matrix::new(m, n);

        group.bench_function(format!("{m}x{k} * {k}x{n}"), |bencher| {
            bencher.iter(|| {
                mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));
            });
        });
    }

    group.finish();
}

fn bench_large_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_matmul");

    // MNIST-like GEMM
    let a = Matrix::new(60000, 784);
    let b = Matrix::new(784, 128);
    let mut out = Matrix::new(60000, 128);

    group.bench_function("60000x784 * 784x128", |bencher| {
        bencher.iter(|| {
            mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));
        });
    });

    group.finish();
}

fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv");

    let a = Matrix::new(60000, 784);
    let b = Matrix::new(784, 1);
    let mut out = Matrix::new(60000, 1);

    group.bench_function("60000x784 * 784x1", |bencher| {
        bencher.iter(|| {
            mat_mul(&mut out, &a, &b, B8(1), B8(0), B8(0));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_small_matmul,
    bench_medium_matmul,
    bench_large_matmul,
    bench_gemv
);

criterion_main!(benches);

