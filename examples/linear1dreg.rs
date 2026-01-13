use mlrs::{Graph, Matrix};

fn main() {
    // y = 2x + 1

    let mut x = Matrix::new(4, 1);
    x.data = vec![1.0, 2.0, 3.0, 4.0];

    let mut y = Matrix::new(4, 1);
    y.data = vec![3.0, 5.0, 7.0, 9.0];

    let mut g = Graph { nodes: vec![] };

    // constants
    let x_id = g.tensor(&x, false);
    let y_id = g.tensor(&y, false);

    // parameters
    let w = g.tensor(&Matrix::new(1, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    g.nodes[w].data.data[0] = 0.0;
    g.nodes[b].data.data[0] = 0.0;

    for _ in 0..1000 {
        g.zero_grad();

        let y_hat = g.matmul(x_id, w); // XW
        let y_hat = g.add(y_hat, b); // XW + b
        let diff = g.sub(y_hat, y_id); // error
        let sq = g.mul(diff, diff); // squared error
        let loss = g.mean(sq); // scalar loss

        g.backtrack(loss); // IMPORTANT
        g.step(0.01);
    }

    println!("w = {}", g.nodes[w].data.data[0]);
    println!("b = {}", g.nodes[b].data.data[0]);
}

