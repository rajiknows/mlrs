use mlrs::{backends::cpu::CPUBackend, graph::Graph};
use std::{fs::File, io::Read};

fn load_data(rows: usize, cols: usize, path: &str) -> (Vec<f32>, Vec<usize>) {
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();

    let floats = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) };

    let data_len = rows * cols;
    let mut data = vec![0.0; data_len];
    let len = floats.len().min(data_len);
    data[..len].copy_from_slice(&floats[..len]);

    (data, vec![rows, cols])
}

fn main() {
    let (train_images_data, train_images_shape) = load_data(60000, 784, "train_images.mat");
    let (train_labels_data, train_labels_shape) = load_data(60000, 1, "train_labels.mat");
    let (test_images_data, test_images_shape) = load_data(10000, 784, "test_images.mat");
    let (test_labels_data, test_labels_shape) = load_data(10000, 1, "test_lables.mat");

    let n = train_images_shape[0];
    let mut g: Graph<f32, CPUBackend> = Graph::new();

    // Data tensors
    let x_id = g.tensor(train_images_data, train_images_shape, false);
    let y_id = g.tensor(train_labels_data, train_labels_shape, false);

    // Parameters
    let w = g.tensor(vec![0.0; 784], vec![784, 1], true);
    let b = g.tensor(vec![0.0], vec![1, 1], true);

    // Xavier initialization for better convergence
    let init_scale = (2.0 / 784.0_f32).sqrt();
    for i in 0..g.nodes[w].inner.data.data.len() {
        g.nodes[w].inner.data.data[i] = (rand::random::<f32>() - 0.5) * init_scale;
    }
    g.nodes[b].inner.data.data[0] = 0.0;

    // Create "one" vector once, outside the loop
    let one = g.tensor(vec![1.0; n], vec![n, 1], false);

    let lr = 0.01;

    for epoch in 0..50 {
        g.zero_grad();

        // Forward pass
        let z = g.matmul(x_id, w); // (60000, 1)
        let z = g.add_broadcast(z, b);
        let y_hat = g.sigmoid(z); // (60000, 1)

        // BCE loss: -mean[y*log(ŷ) + (1-y)*log(1-ŷ)]
        let y_hat_log = g.log(y_hat);
        let one_minus_yhat = g.sub(one, y_hat);
        let one_minus_y = g.sub(one, y_id);
        let log_one_minus_yhat = g.log(one_minus_yhat);

        let term1 = g.mul(y_id, y_hat_log);
        let term2 = g.mul(one_minus_y, log_one_minus_yhat);
        let bce = g.add(term1, term2);
        let neg_bce = g.neg(bce);
        let loss = g.mean(neg_bce);

        // Backward + update
        g.backtrack(loss);
        g.step(lr);

        // Calculate training accuracy every epoch
        if epoch % 5 == 0 || epoch == 49 {
            let loss_val = g.nodes[loss].inner.data.data[0];

            // Calculate accuracy
            let predictions = &g.nodes[y_hat].inner.data;
            let mut correct = 0;
            for i in 0..n {
                let pred_class = if predictions.data[i] > 0.5 { 1.0 } else { 0.0 };
                if pred_class == g.nodes[y_id].inner.data.data[i] {
                    correct += 1;
                }
            }
            let accuracy = 100.0 * correct as f32 / n as f32;

            println!(
                "Epoch {}: loss = {:.4}, accuracy = {:.2}%",
                epoch, loss_val, accuracy
            );
        }
    }

    // Evaluate on test set
    println!("\n=== Test Set Evaluation ===");

    // Manual forward pass on test data
    let w_data = &g.nodes[w].inner.data;
    let b_val = g.nodes[b].inner.data.data[0];

    let mut correct = 0;
    for i in 0..test_images_shape[0] {
        // Compute z = x * w + b
        let mut z = 0.0;
        for j in 0..784 {
            z += test_images_data[i * 784 + j] * w_data.data[j];
        }
        z += b_val;

        // Sigmoid
        let pred_prob = 1.0 / (1.0 + (-z).exp());
        let pred_class = if pred_prob > 0.5 { 1.0 } else { 0.0 };

        if pred_class == test_labels_data[i] {
            correct += 1;
        }
    }

    let test_accuracy = 100.0 * correct as f32 / test_images_shape[0] as f32;
    println!("Test Accuracy: {:.2}%", test_accuracy);

    println!("\nFirst 10 weights: {:?}", &w_data.data[..10]);
    println!("Bias: {:.4}", b_val);
}
