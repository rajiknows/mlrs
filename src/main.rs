use crate::matrix::{Matrix, load_mat};
pub mod gradient;
pub mod matrix;
pub mod model;
pub mod tensor;
pub mod types;

fn draw_mnist_digit_color(m: &Matrix, idx: usize) {
    let start = idx * 28 * 28;

    for y in 0..28 {
        for x in 0..28 {
            let v = m.data[start + y * 28 + x];
            let col: u32 = 232 + (v * 24.0) as u32;
            print!("\x1b[48;5;{}m ", col);
        }
        print!("\x1b[0m\n");
    }
}

fn main() {
    println!("Hello, world!");
    let train_images = load_mat(60000, 784, "train_images.mat");
    let test_images = load_mat(60000, 784, "test_images.mat");
    let mut train_labels = Matrix::new(60000, 10);
    let mut test_labels = Matrix::new(60000, 10);
    {
        let train_label_file = load_mat(60000, 1, "train_labels.mat");
        let test_label_file = load_mat(60000, 1, "test_lables.mat");
        for i in 0..60000 {
            let num = train_label_file.data[i].round() as usize;
            train_labels.data[i * 10 + num] = 1.0f32
        }

        for i in 0..60000 {
            let num = test_label_file.data[i].round() as usize;
            test_labels.data[i * 10 + num] = 1.0f32;
        }
    }

    let idx = 0;
    let start = idx * 784;
    // draw_mnist_digit(&train_images.data[start..start + 784]);

    // println!("{:?}", &train_images.data[0..200]);
    let idx = 50;
    draw_mnist_digit_color(&train_images, idx);

    for i in 0..10 {
        print!("{}", train_labels.data[idx * 10 + i] as u8);
    }
    println!();
}
