use std::sync::Arc;

use crate::{
    backends::Backend,
    numeric::Numeric,
    ops::{self, Operation},
};

pub type TensorId = usize;

pub struct Tensor<T: Numeric, B: Backend> {
    pub id: TensorId,
    pub data: Vec<T>,
    pub grad: Vec<T>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub op: Arc<dyn Operation<T, B>>,
    pub parents: Vec<TensorId>,
    pub req_grad: bool,
}

impl<T: Numeric, B: Backend<DType = T>> Tensor<T, B> {
    pub fn new(
        id: TensorId,
        data: Vec<T>,
        shape: Vec<usize>,
        op: Arc<dyn Operation<T, B>>,
        parents: Vec<TensorId>,
        req_grad: bool,
    ) -> Self {
        let strides = Self::calculate_strides(&shape);
        let data_len = data.len();
        Self {
            id,
            data,
            grad: vec![T::default(); data_len],
            shape,
            stride: strides,
            op,
            parents,
            req_grad,
        }
    }

    pub fn leaf(id: TensorId, data: Vec<T>, shape: Vec<usize>, req_grad: bool) -> Self {
        let strides = Self::calculate_strides(&shape);
        let data_len = data.len();
        Self {
            id,
            data,
            grad: vec![T::default(); data_len],
            op: Arc::new(ops::none::None),
            parents: Vec::new(),
            shape,
            stride: strides,
            req_grad,
        }
    }

    pub fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut acc = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }

        strides
    }
}

pub struct Graph<T: Numeric, B: Backend<DType = T>> {
    pub nodes: Vec<Tensor<T, B>>,
}

impl<T: Numeric, B: Backend<DType = T>> Graph<T, B> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn tensor(&mut self, data: Vec<T>, shape: Vec<usize>, required_grad: bool) -> TensorId {
        let id = self.nodes.len();
        self.nodes
            .push(Tensor::leaf(id, data, shape, required_grad));
        id
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (node_a, node_b) = (&self.nodes[a], &self.nodes[b]);
        let new_tensor = B::add(node_a, node_b);
        let id = self.nodes.len();
        self.nodes.push(Tensor::new(
            id,
            new_tensor.data,
            new_tensor.shape,
            Arc::new(ops::add::Add),
            vec![a, b],
            true,
        ));
        id
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (node_a, node_b) = (&self.nodes[a], &self.nodes[b]);
        let new_tensor = B::sub(node_a, node_b);
        let id = self.nodes.len();
        self.nodes.push(Tensor::new(
            id,
            new_tensor.data,
            new_tensor.shape,
            Arc::new(ops::sub::Sub),
            vec![a, b],
            true,
        ));
        id
    }

    pub fn add_broadcast(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let ma_shape = &self.nodes[a].shape;
        let mb_shape = &self.nodes[b].shape;
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        // Check if b is a scalar (1x1)
        if ma_shape[0] == 1 && mb_shape[1] == 1 {
            let scalar = mb[0];
            let out = ma.iter().map(|x| *x * scalar).collect();

            let id = self.nodes.len();
            self.nodes.push(Tensor::new(
                id,
                out,
                ma_shape.to_owned(),
                Arc::new(ops::add_broadcast::AddBroadbast),
                vec![a, b],
                true,
            ));
            return id;
        }

        // Otherwise, use regular add
        self.add(a, b)
    }

    pub fn neg(&mut self, x: TensorId) -> TensorId {
        let out = B::neg(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::neg::Neg),
            vec![x],
            true,
        ));
        id
    }

    // pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
    //     let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
    //
    //     let mut out = Matrix::new(ma.row, ma.col);
    //     mat_sub(&mut out, ma, mb);
    //
    //     let id = self.nodes.len();
    //     self.nodes.push(Tensor {
    //         id,
    //         data: out,
    //         grad: Matrix::new(ma.row, ma.col),
    //         op: Op::Sub,
    //         parents: vec![a, b],
    //         req_grad: true,
    //     });
    //
    //     id
    // }

    pub fn log(&mut self, x: TensorId) -> TensorId {
        let out = B::log(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::log::Log),
            vec![x],
            true,
        ));
        id
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = B::mul(&self.nodes[a], &self.nodes[b]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::mul::Mul),
            vec![a, b],
            true,
        ));
        id
    }
    pub fn relu(&mut self, x: TensorId) -> TensorId {
        let m = &self.nodes[x].data;

        let mut out = m.clone();
        out.relu();

        let id = self.nodes.len();
        self.nodes.push(Tensor {
            id,
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
            id,
            data: out,
            grad: Matrix::new(m.row, m.col),
            op: Op::SoftMax,
            parents: vec![x],
            req_grad: true,
        });

        id
    }

    // pub fn sum(&mut self, x: TensorId) -> TensorId {
    //     let m = &self.nodes[x].data;
    //
    //     let mut out = Matrix::new(1, 1);
    //     out.data[0] = m.sum();
    //
    //     let id = self.nodes.len();
    //     self.nodes.push(Tensor {
    //         id,
    //         data: out,
    //         grad: Matrix::new(1, 1),
    //         op: Op::Sum,
    //         parents: vec![x],
    //         req_grad: true,
    //     });
    //
    //     id
    // }
    //
    // pub fn mul_scalar(&mut self, input_id: TensorId, x: f32) -> TensorId {
    //     let ma = &self.nodes[input_id].data;
    //     let mut out = Matrix::new(ma.row, ma.col);
    //     for i in 0..out.data.len() {
    //         out.data[i] = ma.data[i] * x;
    //     }
    //     let id = self.nodes.len();
    //     self.nodes.push(Tensor {
    //         id,
    //         data: out,
    //         grad: Matrix::new(ma.row, ma.col),
    //         op: Op::None,
    //         parents: vec![input_id],
    //         req_grad: false,
    //     });
    //     id
    // }

    pub fn mean(&mut self, x: TensorId) -> TensorId {
        let out = B::mean(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::mean::Mean),
            vec![x],
            true,
        ));
        id
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = B::matmul(&self.nodes[a], &self.nodes[b]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::matmul::MatMul),
            vec![a, b],
            true,
        ));
        id
    }

    pub fn sigmoid(&mut self, x: TensorId) -> TensorId {
        let out = B::sigmoid(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data,
            out.shape,
            Arc::new(ops::sigmoid::Sigmoid),
            vec![x],
            true,
        ));
        id
    }

    pub fn backtrack(&mut self, loss: TensorId) {
        for n in &mut self.nodes {
            n.grad.fill(0.0);
        }

        // seed dL/dL = 1
        self.nodes[loss].grad.data[0] = 1.0;

        let topo = self.topo_from(loss);

        // reverse topo: loss → leaves
        for &id in topo.iter().rev() {
            self.backward_node(id);
        }
    }

    fn backward_node(&mut self, id: TensorId) {
        let (op, parents, grad, out_data) = {
            let n = &self.nodes[id];
            (n.op, n.parents.clone(), n.grad.clone(), n.data.clone())
        };

        match op {
            Op::MatMul => {
                let [a, b] = parents[..] else { return };

                let grad_z = &grad;
                let a_data = &self.nodes[a].data;
                let b_data = &self.nodes[b].data;

                // dA = dZ · Bᵀ
                let mut da = Matrix::new(a_data.row, a_data.col);
                mat_mul(&mut da, grad_z, b_data, B8(1), B8(0), B8(1));

                // dB = Aᵀ · dZ
                let mut db = Matrix::new(b_data.row, b_data.col);
                mat_mul(&mut db, a_data, grad_z, B8(1), B8(1), B8(0));

                for i in 0..da.data.len() {
                    self.nodes[a].grad.data[i] += da.data[i];
                }
                for i in 0..db.data.len() {
                    self.nodes[b].grad.data[i] += db.data[i];
                }
            }
            Op::Sum => {
                let [a] = parents[..] else { return };
                let g = grad.data[0];

                for i in 0..self.nodes[a].grad.data.len() {
                    self.nodes[a].grad.data[i] += g;
                }
            }

            Op::Mean => {
                let [a] = parents[..] else { return };
                let scale = grad.data[0] / (self.nodes[a].data.row * self.nodes[a].data.col) as f32;

                for i in 0..self.nodes[a].grad.data.len() {
                    self.nodes[a].grad.data[i] += scale;
                }
            }

            Op::Neg => {
                let [a] = parents[..] else { return };

                for i in 0..grad.data.len() {
                    self.nodes[a].grad.data[i] -= grad.data[i]; // Derivative of -x is -1
                }
            }

            Op::Add => {
                let [a, b] = parents[..] else { return };

                let ga = &mut self.nodes[a].grad;
                for i in 0..ga.data.len() {
                    ga.data[i] += grad.data[i];
                }

                // Handle scalar broadcasting
                let gb = &mut self.nodes[b].grad;
                if gb.row == 1 && gb.col == 1 {
                    // Sum all gradients for scalar
                    let sum: f32 = grad.data.iter().sum();
                    gb.data[0] += sum;
                } else {
                    for i in 0..gb.data.len() {
                        gb.data[i] += grad.data[i];
                    }
                }
            }

            Op::Sub => {
                let [a, b] = parents[..] else { return };

                for i in 0..grad.data.len() {
                    self.nodes[a].grad.data[i] += grad.data[i];
                    self.nodes[b].grad.data[i] -= grad.data[i];
                }
            }

            Op::Log => {
                let [a] = parents[..] else { return };

                for i in 0..grad.data.len() {
                    let val = self.nodes[a].data.data[i];
                    if val.is_normal() && val > 0.0 {
                        // Check for normal positive values
                        self.nodes[a].grad.data[i] += grad.data[i] / val;
                    } else if val > 0.0 {
                        // Handle denormal numbers
                        self.nodes[a].grad.data[i] += grad.data[i] / f32::max(val, 1e-8);
                    }
                    // For val <= 0, gradient is undefined, so skip
                }
            }

            Op::Mul => {
                let [a, b] = parents[..] else { return };
                for i in 0..grad.data.len() {
                    self.nodes[a].grad.data[i] += grad.data[i] * self.nodes[b].data.data[i];
                    self.nodes[b].grad.data[i] += grad.data[i] * self.nodes[a].data.data[i];
                }
            }

            Op::Relu => {
                let [a] = parents[..] else { return };

                for i in 0..grad.data.len() {
                    if self.nodes[a].data.data[i] > 0.0 {
                        self.nodes[a].grad.data[i] += grad.data[i];
                    }
                }
            }

            Op::Sigmoid => {
                let [a] = parents[..] else { return };
                for i in 0..grad.data.len() {
                    let s = out_data.data[i];
                    self.nodes[a].grad.data[i] += grad.data[i] * s * (1.0 - s);
                }
            }

            _ => {}
        }
    }

    pub fn zero_grad(&mut self) {
        for n in &mut self.nodes {
            n.grad.fill(0.0);
        }
    }

    pub fn step(&mut self, lr: f32) {
        for n in &mut self.nodes {
            if n.req_grad {
                for i in 0..n.data.data.len() {
                    n.data.data[i] -= lr * n.grad.data[i];
                }
            }
        }
    }

    fn topo_from(&self, start: TensorId) -> Vec<TensorId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = Vec::new();

        fn dfs(g: &Graph, v: TensorId, visited: &mut Vec<bool>, stack: &mut Vec<TensorId>) {
            if visited[v] {
                return;
            }
            visited[v] = true;

            for &p in &g.nodes[v].parents {
                dfs(g, p, visited, stack);
            }

            stack.push(v);
        }

        dfs(self, start, &mut visited, &mut stack);
        stack
    }

    pub fn mul_scaler(&mut self, a: TensorId, x: T) -> TensorId {
        let node = &self.nodes[a];
    }
}

// pub struct Graph {
//     pub nodes: Vec<Tensor>,
// }
//
// impl Graph {
//     pub fn tensor(&mut self, m: &Matrix, required_grad: bool) -> TensorId {
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             req_grad: required_grad,
//             grad: Matrix::new(m.row, m.col),
//             data: m.clone(),
//             op: Op::Leaf,
//             parents: vec![],
//         });
//
//         id
//     }
//
//
//     pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
//         let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
//
//         let mut out = Matrix::new(ma.row, ma.col);
//         assert_eq!(ma.row, mb.row);
//         assert_eq!(ma.col, mb.col);
//         mat_add(&mut out, ma, mb);
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, ma.col),
//             op: Op::Add,
//             parents: vec![a, b],
//             req_grad: true,
//         });
//
//         id
//     }
//
//
//     pub fn add_broadcast(&mut self, a: TensorId, b: TensorId) -> TensorId {
//         let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
//
//         // Check if b is a scalar (1x1)
//         if mb.row == 1 && mb.col == 1 {
//             let scalar = mb.data[0];
//             let mut out = ma.clone();
//             for x in &mut out.data {
//                 *x += scalar;
//             }
//
//             let id = self.nodes.len();
//             self.nodes.push(Tensor {
//                 id,
//                 data: out,
//                 grad: Matrix::new(ma.row, ma.col),
//                 op: Op::Add,
//                 parents: vec![a, b],
//                 req_grad: true,
//             });
//             return id;
//         }
//
//         // Otherwise, use regular add
//         self.add(a, b)
//     }
//
//     pub fn neg(&mut self, x: TensorId) -> TensorId {
//         // get the matrix and negate it and put it back to the graph
//         let matrix = &self.nodes[x].data;
//         let mut neg_mat = Matrix::new(matrix.row, matrix.col);
//         for i in 0..neg_mat.data.len() {
//             neg_mat.data[i] = -matrix.data[i];
//         }
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: neg_mat,
//             grad: Matrix::new(matrix.row, matrix.col),
//             op: Op::Neg,
//             parents: vec![x],
//             req_grad: true,
//         });
//         id
//     }
//
//
//     pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
//         let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
//
//         let mut out = Matrix::new(ma.row, ma.col);
//         mat_sub(&mut out, ma, mb);
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, ma.col),
//             op: Op::Sub,
//             parents: vec![a, b],
//             req_grad: true,
//         });
//
//         id
//     }
//
//
//     pub fn log(&mut self, a: TensorId) -> TensorId {
//         let ma = &self.nodes[a].data;
//
//         let mut out = ma.clone();
//         for v in &mut out.data {
//             *v = v.max(1e-7).ln();
//         }
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, ma.col),
//             op: Op::Log,
//             parents: vec![a],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
//         let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
//         assert_eq!(ma.row * ma.col, mb.row * mb.col);
//
//         let mut out = Matrix::new(ma.row, ma.col);
//         for i in 0..out.data.len() {
//             out.data[i] = ma.data[i] * mb.data[i];
//         }
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, ma.col),
//             op: Op::Mul,
//             parents: vec![a, b],
//             req_grad: true,
//         });
//         id
//     }
//
//     pub fn relu(&mut self, x: TensorId) -> TensorId {
//         let m = &self.nodes[x].data;
//
//         let mut out = m.clone();
//         out.relu();
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(m.row, m.col),
//             op: Op::Relu,
//             parents: vec![x],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn softmax(&mut self, x: TensorId) -> TensorId {
//         let m = &self.nodes[x].data;
//
//         let mut out = m.clone();
//         out.softmax();
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(m.row, m.col),
//             op: Op::SoftMax,
//             parents: vec![x],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn sum(&mut self, x: TensorId) -> TensorId {
//         let m = &self.nodes[x].data;
//
//         let mut out = Matrix::new(1, 1);
//         out.data[0] = m.sum();
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(1, 1),
//             op: Op::Sum,
//             parents: vec![x],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn mul_scalar(&mut self, input_id: TensorId, x: f32) -> TensorId {
//         let ma = &self.nodes[input_id].data;
//         let mut out = Matrix::new(ma.row, ma.col);
//         for i in 0..out.data.len() {
//             out.data[i] = ma.data[i] * x;
//         }
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, ma.col),
//             op: Op::None,
//             parents: vec![input_id],
//             req_grad: false,
//         });
//         id
//     }
//
//     pub fn mean(&mut self, x: TensorId) -> TensorId {
//         let m = &self.nodes[x].data;
//
//         let mut out = Matrix::new(1, 1);
//         out.data[0] = m.sum() / (m.row * m.col) as f32;
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(1, 1),
//             op: Op::Mean,
//             parents: vec![x],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
//         let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);
//
//         let mut out = Matrix::new(ma.row, mb.col);
//         mat_mul(&mut out, ma, mb, B8(1), B8(0), B8(0));
//
//         let id = self.nodes.len();
//         self.nodes.push(Tensor {
//             id,
//             data: out,
//             grad: Matrix::new(ma.row, mb.col),
//             op: Op::MatMul,
//             parents: vec![a, b],
//             req_grad: true,
//         });
//
//         id
//     }
//
//     pub fn sigmoid(&mut self, z: TensorId) -> TensorId {
//         let tensor = &mut self.nodes[z];
//         let matrix_clone = &mut tensor.data.clone();
//         let id = self.nodes.len();
//
//         // calculate the sigmoid of matrix
//         matrix_clone.sigmoid();
//
//         self.nodes.push(Tensor {
//             id,
//             data: matrix_clone.clone(),
//             grad: Matrix::new(matrix_clone.row, matrix_clone.col),
//             op: Op::Sigmoid,
//             parents: vec![z],
//             req_grad: true,
//         });
//         id
//     }
//
//     pub fn backtrack(&mut self, loss: TensorId) {
//         for n in &mut self.nodes {
//             n.grad.fill(0.0);
//         }
//
//         // seed dL/dL = 1
//         self.nodes[loss].grad.data[0] = 1.0;
//
//         let topo = self.topo_from(loss);
//
//         // reverse topo: loss → leaves
//         for &id in topo.iter().rev() {
//             self.backward_node(id);
//         }
//     }
//
//     fn backward_node(&mut self, id: TensorId) {
//         let (op, parents, grad, out_data) = {
//             let n = &self.nodes[id];
//             (n.op, n.parents.clone(), n.grad.clone(), n.data.clone())
//         };
//
//         match op {
//             Op::MatMul => {
//                 let [a, b] = parents[..] else { return };
//
//                 let grad_z = &grad;
//                 let a_data = &self.nodes[a].data;
//                 let b_data = &self.nodes[b].data;
//
//                 // dA = dZ · Bᵀ
//                 let mut da = Matrix::new(a_data.row, a_data.col);
//                 mat_mul(&mut da, grad_z, b_data, B8(1), B8(0), B8(1));
//
//                 // dB = Aᵀ · dZ
//                 let mut db = Matrix::new(b_data.row, b_data.col);
//                 mat_mul(&mut db, a_data, grad_z, B8(1), B8(1), B8(0));
//
//                 for i in 0..da.data.len() {
//                     self.nodes[a].grad.data[i] += da.data[i];
//                 }
//                 for i in 0..db.data.len() {
//                     self.nodes[b].grad.data[i] += db.data[i];
//                 }
//             }
//             Op::Sum => {
//                 let [a] = parents[..] else { return };
//                 let g = grad.data[0];
//
//                 for i in 0..self.nodes[a].grad.data.len() {
//                     self.nodes[a].grad.data[i] += g;
//                 }
//             }
//
//             Op::Mean => {
//                 let [a] = parents[..] else { return };
//                 let scale = grad.data[0] / (self.nodes[a].data.row * self.nodes[a].data.col) as f32;
//
//                 for i in 0..self.nodes[a].grad.data.len() {
//                     self.nodes[a].grad.data[i] += scale;
//                 }
//             }
//
//             Op::Neg => {
//                 let [a] = parents[..] else { return };
//
//                 for i in 0..grad.data.len() {
//                     self.nodes[a].grad.data[i] -= grad.data[i]; // Derivative of -x is -1
//                 }
//             }
//
//             Op::Add => {
//                 let [a, b] = parents[..] else { return };
//
//                 let ga = &mut self.nodes[a].grad;
//                 for i in 0..ga.data.len() {
//                     ga.data[i] += grad.data[i];
//                 }
//
//                 // Handle scalar broadcasting
//                 let gb = &mut self.nodes[b].grad;
//                 if gb.row == 1 && gb.col == 1 {
//                     // Sum all gradients for scalar
//                     let sum: f32 = grad.data.iter().sum();
//                     gb.data[0] += sum;
//                 } else {
//                     for i in 0..gb.data.len() {
//                         gb.data[i] += grad.data[i];
//                     }
//                 }
//             }
//
//             Op::Sub => {
//                 let [a, b] = parents[..] else { return };
//
//                 for i in 0..grad.data.len() {
//                     self.nodes[a].grad.data[i] += grad.data[i];
//                     self.nodes[b].grad.data[i] -= grad.data[i];
//                 }
//             }
//
//             Op::Log => {
//                 let [a] = parents[..] else { return };
//
//                 for i in 0..grad.data.len() {
//                     let val = self.nodes[a].data.data[i];
//                     if val.is_normal() && val > 0.0 {
//                         // Check for normal positive values
//                         self.nodes[a].grad.data[i] += grad.data[i] / val;
//                     } else if val > 0.0 {
//                         // Handle denormal numbers
//                         self.nodes[a].grad.data[i] += grad.data[i] / f32::max(val, 1e-8);
//                     }
//                     // For val <= 0, gradient is undefined, so skip
//                 }
//             }
//
//             Op::Mul => {
//                 let [a, b] = parents[..] else { return };
//                 for i in 0..grad.data.len() {
//                     self.nodes[a].grad.data[i] += grad.data[i] * self.nodes[b].data.data[i];
//                     self.nodes[b].grad.data[i] += grad.data[i] * self.nodes[a].data.data[i];
//                 }
//             }
//
//             Op::Relu => {
//                 let [a] = parents[..] else { return };
//
//                 for i in 0..grad.data.len() {
//                     if self.nodes[a].data.data[i] > 0.0 {
//                         self.nodes[a].grad.data[i] += grad.data[i];
//                     }
//                 }
//             }
//
//             Op::Sigmoid => {
//                 let [a] = parents[..] else { return };
//                 for i in 0..grad.data.len() {
//                     let s = out_data.data[i];
//                     self.nodes[a].grad.data[i] += grad.data[i] * s * (1.0 - s);
//                 }
//             }
//
//             _ => {}
//         }
//     }
//
//     pub fn zero_grad(&mut self) {
//         for n in &mut self.nodes {
//             n.grad.fill(0.0);
//         }
//     }
//
//     pub fn step(&mut self, lr: f32) {
//         for n in &mut self.nodes {
//             if n.req_grad {
//                 for i in 0..n.data.data.len() {
//                     n.data.data[i] -= lr * n.grad.data[i];
//                 }
//             }
//         }
//     }
//
//     fn topo_from(&self, start: TensorId) -> Vec<TensorId> {
//         let mut visited = vec![false; self.nodes.len()];
//         let mut stack = Vec::new();
//
//         fn dfs(g: &Graph, v: TensorId, visited: &mut Vec<bool>, stack: &mut Vec<TensorId>) {
//             if visited[v] {
//                 return;
//             }
//             visited[v] = true;
//
//             for &p in &g.nodes[v].parents {
//                 dfs(g, p, visited, stack);
//             }
//
//             stack.push(v);
//         }
//
//         dfs(self, start, &mut visited, &mut stack);
//         stack
//     }
// }
//
//
// fn topo_sort(adj_mat: &Vec<Vec<TensorId>>) -> Vec<TensorId> {
//     let n = adj_mat.len();
//     let mut stack: VecDeque<TensorId> = VecDeque::new();
//     let mut visited = vec![false; n];
//
//     for i in 0..n {
//         if !visited[i] {
//             find_topo_sort(i, &mut visited, adj_mat, &mut stack);
//         }
//     }
//     let mut topo = Vec::new();
//
//     while !stack.is_empty() {
//         if let Some(el) = stack.pop_back() {
//             topo.push(el);
//         }
//     }
//     topo
// }
//
// fn find_topo_sort(
//     i: usize,
//     visited: &mut Vec<bool>,
//     adj_mat: &Vec<Vec<TensorId>>,
//     stack: &mut VecDeque<TensorId>,
// ) {
//     visited[i] = true;
//
//     for &node in &adj_mat[i] {
//         if !visited[node] {
//             find_topo_sort(node, visited, adj_mat, stack);
//         }
//     }
//
//     stack.push_back(i);
// }
//
//
// pub fn mse(y_hat: &Matrix, y: &Matrix) -> f32 {
//     let size = y_hat.data.len();
//     if size == 0 {
//         return 0.0;
//     }
//     assert_eq!(y_hat.data.len(), y.data.len());
//     let mut mse = 0.0;
//     for i in 0..size {
//         mse += (y_hat.data[i] - y.data[i]).powi(2);
//     }
//     mse / size as f32
// }
