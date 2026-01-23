use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mlrs::matrix::{Matrix, mat_add};

fn bench_add(c: &mut Criterion) {
    let a = Matrix::new(60000, 1);
    let b = Matrix::new(60000, 1);
    let mut out = Matrix::new(60000, 1);

    c.bench_function("add 60000x1", |bencher| {
        bencher.iter(|| {
            mat_add(black_box(&mut out), black_box(&a), black_box(&b));
        });
    });
}

criterion_group!(benches, bench_add);
criterion_main!(benches);
