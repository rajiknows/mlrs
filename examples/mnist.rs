use mlrs::{Graph, Matrix, matrix::load_mat};

fn main() {
    // load MNIST (0 vs 1)
    let train_images = load_mat(60000, 784, "train_images.mat");
    let test_images = load_mat(60000, 784, "test_images.mat");
    let train_labels = load_mat(60000, 1, "train_labels.mat");

    // let (train_images, train_labels) =
    //     mnist_load_binary("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    //
    let n = train_images.row; // number of samples

    let mut g = Graph { nodes: vec![] };

    // data tensor
    let x_id = g.tensor(&train_images, false);
    let y_id = g.tensor(&train_labels, false);

    // parameters: weights (784×1), bias (1×1)
    let w = g.tensor(&Matrix::new(784, 1), true);
    let b = g.tensor(&Matrix::new(1, 1), true);

    // init
    g.nodes[w].data.data.fill(0.0);
    g.nodes[b].data.data[0] = 0.0;

    let lr = 0.1;

    for epoch in 0..50 {
        g.zero_grad();

        // forward
        let z = g.matmul(x_id, w); // (N×1)
        let z = g.add(z, b); // (N×1)
        let y_hat = g.sigmoid(z); // (N×1)

        // BCE loss
        // loss per element: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        let one = {
            let mut m = Matrix::new(n, 1);
            m.data.fill(1.0);
            g.tensor(&m, false)
        };

        let y_hat_log = g.log(y_hat);
        let one_minus_yhat = g.sub(one, y_hat);
        let one_minus_y = g.sub(one, y_id);
        let log_one_minus_yhat = g.log(one_minus_yhat);

        let term1 = g.mul(y_id, y_hat_log);
        let term2 = g.mul(one_minus_y, log_one_minus_yhat);
        let bce = g.add(term1, term2);
        let neg_bce = g.neg(bce);
        let loss = g.mean(neg_bce); // neg = new op or multiply by -1

        // backward + update
        g.backtrack(loss);
        g.step(lr);

        println!("epoch {}: loss = {}", epoch, g.nodes[loss].data.data[0]);
    }

    // evaluate
    let out = g.nodes[w].data.clone();
    let bias = g.nodes[b].data.data[0];
    println!("weights: {:?} bias {}", out.data[..10].to_vec(), bias);
}
