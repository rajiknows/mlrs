use crate::{
    matrix::{Matrix, mat_add, mat_mul, mat_sub},
    types::B8,
};

#[derive(Clone, Copy)]
pub enum Op {
    None,
    Leaf,
    Add,
    Sub,
    Mul,
    Relu,
    SoftMax,
    MatMul,
}

pub type TensorId = usize;

pub struct Tensor {
    pub data: Matrix,
    pub grad: Matrix,
    pub op: Op,
    pub parents: Vec<TensorId>,
    pub req_grad: bool,
}

impl Tensor {
    pub fn leaf(matrix: &Matrix, grad: &Matrix) -> Self {
        Self {
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
            data: out,
            grad: Matrix::new(ma.row, mb.col),
            op: Op::MatMul,
            parents: vec![a, b],
            req_grad: true,
        });

        id
    }
}
