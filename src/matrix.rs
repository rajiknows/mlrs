use std::{fs::File, io::Read};

use crate::types::{B8, B32};

#[warn(dead_code)]
#[derive(Debug, Clone)]
pub struct Matrix {
    pub row: usize,
    pub col: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            data: vec![0.0; row * col],
        }
    }

    pub fn mat_copy(&self, dst: &mut Matrix) -> B32 {
        if self.row != dst.row || self.col != dst.col {
            return B32(0);
        }
        dst.data.copy_from_slice(&self.data);
        B32(1)
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    pub fn fill(&mut self, x: f32) {
        self.data.fill(x);
    }

    pub fn scale(&mut self, scale: f32) {
        let size = self.row * self.col;
        for i in 0..size {
            self.data[i] *= scale;
        }
    }

    pub fn sum(&self) -> f32 {
        let mut sum: f32 = 0.00;
        let size = self.row * self.col;
        for i in 0..size {
            sum += self.data[i]
        }
        sum
    }

    pub fn sigmoid(&self) {
        todo!()
    }

    // this is literally just max(o,x)
    pub fn relu(&mut self) -> B32 {
        let size = self.row * self.col;
        for i in 0..size {
            self.data[i] = self.data[i].max(0.0);
        }
        B32(1)
    }

    pub fn softmax(&mut self) -> B32 {
        // o_i = e^a_i / sum(e^a_i)
        let mut sum = 0.0f32;
        let size = self.row * self.col;
        for i in 0..size {
            self.data[i] = self.data[i].exp();
            sum += self.data[i];
        }

        self.scale(1.0f32 / sum);
        B32(1)
    }

    pub fn cross_entropy(&mut self, p: &Matrix, q: &Matrix) -> B32 {
        if p.row != q.row || p.col != q.col {
            return B32(0);
        }
        if self.row != p.row || self.col != p.col {
            return B32(0);
        }

        let size = self.row * self.col;
        for i in 0..size {
            self.data[i] = if p.data[i] == 0.0 {
                0.0
            } else {
                p.data[i] * -q.data[i].ln()
            }
        }

        B32(1)
    }
    // fn relu_add_grad(&mut self, p: &Matrix) -> B32 {
    //     todo!()
    // }
    // fn softmax_add_grad(&mut self, p: &Matrix) -> B32 {
    //     todo!()
    // }
    // fn cross_entropy_add_grad(&mut self, p: &Matrix, q: &Matrix) {}
}

pub fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> B32 {
    if a.row != b.row || a.col != b.col {
        return B32(0);
    }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    B32(1)
}

pub fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> B32 {
    if a.row != b.row || a.col != b.col {
        return B32(0);
    }
    if out.row != a.row || out.col != a.col {
        return B32(0);
    }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] - b.data[i];
    }
    B32(1)
}

pub fn mat_mul(
    out: &mut Matrix,
    a: &Matrix,
    b: &Matrix,
    zero_out: B8,
    transpose_a: B8,
    transpose_b: B8,
) -> B32 {
    let a_rows = if transpose_a.get() { a.col } else { a.row };
    let a_cols = if transpose_a.get() { a.row } else { a.col };
    let b_rows = if transpose_b.get() { b.col } else { b.row };
    let b_cols = if transpose_b.get() { b.row } else { b.col };

    if a_cols != b_rows {
        return B32(0);
    }
    if out.row != a_rows || out.col != b_cols {
        return B32(0);
    }

    if zero_out.get() {
        out.clear();
    }

    // so now we got 4 cases
    //                      a       b
    //  transpose?          t       n
    //                      t       t
    //                      n       t
    //                      n       n
    let tranpose = (transpose_a.0 << 1) | transpose_b.0;
    match tranpose {
        0x00 => {
            _mat_mul_nn(out, a, b);
        }
        0x01 => {
            _mat_mul_nt(out, a, b);
        }
        0x10 => {
            _mat_mul_tn(out, a, b);
        }
        0x11 => {
            _mat_mul_tt(out, a, b);
        }
        _ => {}
    }

    return B32(1);
}

fn _mat_mul_nn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.row {
        for j in 0..out.col {
            let mut sum = 0.0;
            for k in 0..a.col {
                sum += a.data[i * a.col + k] * b.data[k * b.col + j];
            }
            out.data[i * out.col + j] += sum;
        }
    }
}

fn _mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.row {
        for j in 0..out.col {
            for k in 0..a.col {
                out.data[j + i * out.col] += a.data[k + i * a.col] + b.data[k + j * b.col];
            }
        }
    }
}

fn _mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k in 0..a.row {
        for i in 0..out.row {
            for j in 0..out.col {
                out.data[j + i * out.col] += a.data[i + k * a.col] + b.data[j + k * b.col];
            }
        }
    }
}

fn _mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i in 0..out.row {
        for j in 0..out.col {
            for k in 0..a.col {
                out.data[j + i * out.col] += a.data[i + k * a.col] + b.data[k + j * b.col];
            }
        }
    }
}

pub fn load_mat(rows: usize, cols: usize, path: &str) -> Matrix {
    let mut m = Matrix::new(rows, cols);

    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();

    let floats = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) };

    let len = floats.len().min(m.data.len());
    m.data[..len].copy_from_slice(&floats[..len]);
    m
}
