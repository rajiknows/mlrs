use std::collections::VecDeque;

use crate::{
    matrix::{Matrix, mat_add, mat_mul, mat_sub},
    types::B8,
};

#[derive(Clone, Copy)]
pub enum Op {
    None,
    Leaf,
    Sigmoid,
    Add,
    Sub,
    Mul,
    Relu,
    SoftMax,
    MatMul,
}

pub type TensorId = usize;

pub struct Tensor {
    pub id: TensorId,
    pub data: Matrix,
    pub grad: Matrix,
    pub op: Op,
    pub parents: Vec<TensorId>,
    pub req_grad: bool,
}

impl Tensor {
    pub fn leaf(matrix: &Matrix, grad: &Matrix) -> Self {
        Self {
            id: 0,
            data: matrix.clone(),
            grad: grad.clone(),
            op: Op::None,
            parents: Vec::new(),
            req_grad: true,
        }
    }
}

pub struct Graph {
    nodes: Vec<Tensor>,
}

impl Graph {
    pub fn tensor(&mut self, m: &Matrix, required_grad: bool) -> TensorId {
        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            req_grad: required_grad,
            grad: Matrix::new(m.row, m.col),
            data: m.clone(),
            op: Op::Leaf,
            parents: vec![],
        });

        id
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        let mut out = Matrix::new(ma.row, ma.col);
        mat_add(&mut out, ma, mb);

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(ma.row, ma.col),
            op: Op::Add,
            parents: vec![a, b],
            req_grad: true,
        });

        id
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        let mut out = Matrix::new(ma.row, ma.col);
        mat_sub(&mut out, ma, mb);

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(ma.row, ma.col),
            op: Op::Sub,
            parents: vec![a, b],
            req_grad: true,
        });

        id
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        let mut out = Matrix::new(ma.row, ma.col);
        mat_mul(&mut out, ma, mb, B8(1), B8(0), B8(0));

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(ma.row, ma.col),
            op: Op::Mul,
            parents: vec![a, b],
            req_grad: true,
        });

        id
    }

    pub fn relu(&mut self, x: TensorId) -> TensorId {
        let m = &self.nodes[x].data;

        let mut out = m.clone();
        out.relu();

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(m.row, m.col),
            op: Op::Relu,
            parents: vec![x],
            req_grad: true,
        });

        id
    }

    pub fn softmax(&mut self, x: TensorId) -> TensorId {
        let m = &self.nodes[x].data;

        let mut out = m.clone();
        out.softmax();

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(m.row, m.col),
            op: Op::SoftMax,
            parents: vec![x],
            req_grad: true,
        });

        id
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        let mut out = Matrix::new(ma.row, mb.col);
        mat_mul(&mut out, ma, mb, B8(1), B8(0), B8(0));

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id: id,
            data: out,
            grad: Matrix::new(ma.row, mb.col),
            op: Op::MatMul,
            parents: vec![a, b],
            req_grad: true,
        });

        id
    }

    pub fn sigmoid(&mut self, z: TensorId) -> TensorId {
        let tensor = &self.nodes[z];
        let matrix_clone = &tensor.data.clone();
        let id = self.nodes.len();

        // calculate the sigmoid of matrix
        matrix_clone.sigmoid();

        self.nodes.push(Tensor {
            id: id,
            data: matrix_clone.clone(),
            grad: Matrix::new(matrix_clone.row, matrix_clone.col),
            op: Op::Sigmoid,
            parents: vec![z],
            req_grad: true,
        });
        id
    }

    pub fn step(&mut self, lr: f32) {
        todo!()
    }

    fn backtrack(&mut self) {
        // first we need to make the adjacency list for this
        //
        // adj[][] = [[TensorId]]
        let n = self.nodes.len();
        let mut adj_list: Vec<Vec<TensorId>> = vec![Vec::new(); n];

        for node in &self.nodes {
            for &parent in &node.parents {
                adj_list[parent].push(node.id);
            }
        }
        let topo_sorted = topo_sort(&adj_list);

        // now we need to find the gradient of the tensors in this order
        // and fill them in the tensro
        //

        let last = self.nodes.len() - 1;
        self.nodes[last].grad.fill(1.0);

        for &tid in topo_sorted.iter().rev() {
            self.backward_node(tid);
        }

        todo!()
    }

    fn backward_node(&mut self, id: TensorId) {
        let grad = self.nodes[id].grad.clone();

        match self.nodes[id].op {
            Op::Add => {
                let [a, b] = self.nodes[id].parents[..] else {
                    return;
                };
                let mut a_matrix = Matrix::new(self.nodes[a].data.row, self.nodes[a].data.col);

                let mut b_matrix = Matrix::new(self.nodes[b].data.row, self.nodes[b].data.col);

                mat_add(&mut a_matrix, &self.nodes[a].data, &grad);
                mat_add(&mut b_matrix, &self.nodes[b].data, &grad);
                self.nodes[a].data = a_matrix;
                self.nodes[b].grad = b_matrix;
            }
            Op::Sub => {
                let [a, b] = self.nodes[id].parents[..] else {
                    return;
                };
                let mut a_matrix = Matrix::new(self.nodes[a].data.row, self.nodes[a].data.col);

                let mut b_matrix = Matrix::new(self.nodes[b].data.row, self.nodes[b].data.col);

                mat_sub(&mut a_matrix, &self.nodes[a].data, &grad);
                mat_sub(&mut b_matrix, &self.nodes[b].data, &grad);
                self.nodes[a].data = a_matrix;
                self.nodes[b].grad = b_matrix;
            }
            Op::Mul => {
                let [a, b] = self.nodes[id].parents[..] else {
                    return;
                };
                let mut a_matrix = Matrix::new(self.nodes[a].data.row, self.nodes[a].data.col);

                let mut b_matrix = Matrix::new(self.nodes[b].data.row, self.nodes[b].data.col);

                mat_mul(&mut a_matrix, &self.nodes[a].data, &grad);
                mat_mul(&mut b_matrix, &self.nodes[b].data, &grad);
                self.nodes[a].data = a_matrix;
                self.nodes[b].grad = b_matrix;
            }
            _ => {}
        }
    }
}

fn topo_sort(adj_mat: &Vec<Vec<TensorId>>) -> Vec<TensorId> {
    let n = adj_mat.len();
    let mut stack: VecDeque<TensorId> = VecDeque::new();
    let mut visited = vec![false; n];

    for i in 0..n {
        if !visited[i] {
            find_topo_sort(i, &mut visited, &adj_mat, &mut stack);
        }
    }
    let mut topo = Vec::new();

    while !stack.is_empty() {
        if let Some(el) = stack.pop_back() {
            topo.push(el);
        }
    }
    topo
}

fn find_topo_sort(
    i: usize,
    visited: &mut Vec<bool>,
    adj_mat: &Vec<Vec<TensorId>>,
    stack: &mut VecDeque<TensorId>,
) {
    visited[i] = true;

    for &node in &adj_mat[i] {
        if !visited[node] {
            find_topo_sort(node, visited, adj_mat, stack);
        }
    }

    stack.push_back(i);
}
