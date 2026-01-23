use criterion::{Criterion, criterion_group, criterion_main};

fn bench_forward_backward(c: &mut Criterion) {
    use mlrs::{Graph, Matrix};

    let x = Matrix::new(1024, 784);
    let y = Matrix::new(1024, 1);

    let mut g = Graph { nodes: vec![] };
    let x_id = g.tensor(&x, false);
    let y_id = g.tensor(&y, false);
    let w = g.tensor(&Matrix::new(784, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    c.bench_function("forward+backward", |bencher| {
        bencher.iter(|| {
            g.zero_grad();
            let z = g.matmul(x_id, w);
            let z = g.add_broadcast(z, b);
            let y_hat = g.sigmoid(z);
            let diff = g.sub(y_hat, y_id);
            let sq = g.mul(diff, diff);
            let loss = g.mean(sq);
            g.backtrack(loss);
        });
    });
}

criterion_group!(benches, bench_forward_backward);
criterion_main!(benches);
