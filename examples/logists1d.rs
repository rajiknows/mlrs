use mlrs::{Graph, Matrix};

fn main() {
    let mut x = Matrix::new(6, 1);
    x.data = vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];

    let mut y = Matrix::new(6, 1);
    y.data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut g = Graph { nodes: vec![] };

    let x_id = g.tensor(&x, false);
    let y_id = g.tensor(&y, false);
    let w = g.tensor(&Matrix::new(1, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    g.nodes[w].data.data[0] = 0.5;
    g.nodes[b].data.data[0] = 0.0;

    let one = {
        let mut m = Matrix::new(6, 1);
        m.fill(1.0);
        g.tensor(&m, false)
    };

    for epoch in 0..3000 {
        g.zero_grad();

        let z = g.matmul(x_id, w);
        let z = g.add_broadcast(z, b); // USE add_broadcast!
        let y_hat = g.sigmoid(z);

        // BCE: -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
        let one_minus_y = g.sub(one, y_id);
        let one_minus_y_hat = g.sub(one, y_hat);

        let log_y_hat = g.log(y_hat);
        let log_1_y_hat = g.log(one_minus_y_hat);

        let t1 = g.mul(y_id, log_y_hat);
        let t2 = g.mul(one_minus_y, log_1_y_hat);

        let bce_sum = g.add(t1, t2);
        let bce_mean = g.mean(bce_sum);
        let loss = g.neg(bce_mean);

        g.backtrack(loss);
        g.step(0.1);

        if epoch % 500 == 0 {
            let loss_val = g.nodes[loss].data.data[0];
            println!(
                "Epoch {}: loss = {:.4}, w = {:.4}, b = {:.4}",
                epoch, loss_val, g.nodes[w].data.data[0], g.nodes[b].data.data[0]
            );
        }
    }

    println!("\nFinal parameters:");
    println!("w = {:.4}", g.nodes[w].data.data[0]);
    println!("b = {:.4}", g.nodes[b].data.data[0]);

    println!("\nPredictions:");
    for &xi in &x.data {
        let z = xi * g.nodes[w].data.data[0] + g.nodes[b].data.data[0];
        let p = 1.0 / (1.0 + (-z).exp());
        let class = if p > 0.5 { 1 } else { 0 };
        println!("x = {:>5.1}, y_hat = {:.3}, class = {}", xi, p, class);
    }
}

