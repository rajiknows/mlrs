use mlrs::{Graph, Matrix};

fn main() {
    // Binary classification: y = sigmoid(5x - 2)
    // Data is linearly separable

    let mut x = Matrix::new(6, 1);
    x.data = vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];

    let mut y = Matrix::new(6, 1);
    y.data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut g = Graph { nodes: vec![] };

    let x_id = g.tensor(&x, false);
    let y_id = g.tensor(&y, false);

    let w = g.tensor(&Matrix::new(1, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    g.nodes[w].data.data[0] = 0.0;
    g.nodes[b].data.data[0] = 0.0;

    for _ in 0..2000 {
        g.zero_grad();

        let z = g.matmul(x_id, w); // wx
        let z = g.add(z, b); // wx + b
        let y_hat = g.sigmoid(z); // sigmoid(wx + b)

        let diff = g.sub(y_hat, y_id);
        let sq = g.mul(diff, diff);
        let loss = g.mean(sq); // scalar loss

        g.backtrack(loss);
        g.step(0.1);
    }

    println!("w = {}", g.nodes[w].data.data[0]);
    println!("b = {}", g.nodes[b].data.data[0]);

    // Inference
    println!("Predictions:");
    for i in 0..x.data.len() {
        let v =
            1.0 / (1.0 + (-(x.data[i] * g.nodes[w].data.data[0] + g.nodes[b].data.data[0])).exp());
        println!("x = {:>4}, y_hat = {:.3}", x.data[i], v);
    }
}
