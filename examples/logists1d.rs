use mlrs::{Graph, Matrix};

fn main() {
    // Binary classification: y = step(x)
    // Linearly separable

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

    let one = {
        let mut m = Matrix::new(6, 1);
        m.fill(1.0);
        g.tensor(&m, false)
    };

    for _ in 0..3000 {
        g.zero_grad();

        // z = XW + b
        let z = g.matmul(x_id, w);
        let z = g.add(z, b);

        // ŷ = sigmoid(z)
        let y_hat = g.sigmoid(z);

        // BCE loss:
        // L = -mean(y*log(ŷ) + (1-y)*log(1-ŷ))

        let one_minus_y = g.sub(one, y_id);
        let one_minus_y_hat = g.sub(one, y_hat);

        let log_y_hat = g.log(y_hat); // you need log op
        let log_1_y_hat = g.log(one_minus_y_hat);

        let t1 = g.mul(y_id, log_y_hat);
        let t2 = g.mul(one_minus_y, log_1_y_hat);
        let add = g.add(t1, t2);
        let loss = g.mean(add);
        let loss = g.mul_scalar(loss, -1.0);

        g.backtrack(loss);
        g.step(0.1);
    }

    println!("w = {}", g.nodes[w].data.data[0]);
    println!("b = {}", g.nodes[b].data.data[0]);

    println!("Predictions:");
    for &xi in &x.data {
        let z = xi * g.nodes[w].data.data[0] + g.nodes[b].data.data[0];
        let p = 1.0 / (1.0 + (-z).exp());
        println!("x = {:>4}, y_hat = {:.3}", xi, p);
    }
}

