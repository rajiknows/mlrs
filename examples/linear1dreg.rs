use mlrs::{Graph, Matrix};

fn main() {
    // y = 2x + 1

    let mut x = Matrix::new(4, 1);
    x.data = vec![1.0, 2.0, 3.0, 4.0];

    let mut y = Matrix::new(4, 1);
    y.data = vec![3.0, 5.0, 7.0, 9.0];

    let mut g = Graph { nodes: vec![] };

    let x_id = g.tensor(&x, false);
    let y_id = g.tensor(&y, false);

    let w = g.tensor(&Matrix::new(1, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    g.nodes[w].data.data[0] = 0.0;
    g.nodes[b].data.data[0] = 0.0;

    for _ in 0..1000 {
        g.zero_grad();

        let y_hat = g.matmul(x_id, w);
        let y_hat = g.add(y_hat, b);
        let diff = g.sub(y_hat, y_id);
        let loss = g.mul(diff, diff);

        g.backtrack();
        g.step(0.01);
    }

    println!("w = {}", g.nodes[w].data.data[0]);
    println!("b = {}", g.nodes[b].data.data[0]);
}
