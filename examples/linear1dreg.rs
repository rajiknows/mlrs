use mlrs::{
    backends::cpu::CPUBackend,
    tensor::Graph,
};

fn main() {
    // y = 2x + 1
    let mut g: Graph<f32, CPUBackend> = Graph::new();

    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0];

    let x = g.tensor(x_data, vec![4, 1], false);
    let y = g.tensor(y_data, vec![4, 1], false);

    let w = g.tensor(vec![0.0], vec![1, 1], true);
    let b = g.tensor(vec![0.0], vec![1, 1], true);

    for _ in 0..1000 {
        g.zero_grad();

        let y_hat = g.matmul(x, w);
        let y_hat = g.add_broadcast(y_hat, b);
        let diff = g.sub(y_hat, y);
        let sq = g.mul(diff, diff);
        let loss = g.mean(sq);

        g.backtrack(loss);
        g.step(0.01);
    }

    println!("w = {}", g.nodes[w].data.data[0]);
    println!("b = {}", g.nodes[b].data.data[0]);
}

