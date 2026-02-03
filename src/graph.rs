use std::sync::Arc;

use crate::{
    backends::Backend,
    numeric::Numeric,
    ops,
    tensor::{Tensor, TensorId},
};

pub struct Graph<T: Numeric, B: Backend<DType = T>> {
    pub nodes: Vec<Tensor<T, B>>,
}

impl<T, B> Graph<T, B>
where
    T: Numeric,
    B: Backend<DType = T>,
{
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
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        assert_eq!(node_a.shape, node_b.shape);

        let inner = B::add(&node_a.inner, &node_b.inner, &node_a.shape);

        let id = self.nodes.len();
        let stride = Tensor::<T, B>::calculate_strides(&node_a.shape);

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node_a.shape.clone(),
            stride,
            op: Arc::new(ops::Add),
            parents: vec![a, b],
            req_grad: node_a.req_grad || node_b.req_grad,
        });

        id
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (node_a, node_b) = (&self.nodes[a], &self.nodes[b]);
        let new_tensor = B::sub(node_a, node_b);
        let id = self.nodes.len();
        self.nodes.push(Tensor::new(
            id,
            new_tensor.data.data,
            Arc::new(ops::Sub),
            new_tensor.data.shape,
            vec![a, b],
            true,
        ));
        id
    }

    pub fn add_broadcast(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let ma_shape = &self.nodes[a].data.shape;
        let mb_shape = &self.nodes[b].data.shape;
        let (ma, mb) = (&self.nodes[a].data, &self.nodes[b].data);

        // Check if b is a scalar (1x1)
        if mb_shape[0] == 1 && mb_shape[1] == 1 {
            let scalar = mb.data[0];
            let out = ma.data.iter().map(|x| *x * scalar).collect();

            let id = self.nodes.len();
            self.nodes.push(Tensor::new(
                id,
                out,
                Arc::new(ops::AddBroadcast),
                ma_shape.to_owned(),
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
            out.data.data,
            Arc::new(ops::Neg),
            out.data.shape,
            vec![x],
            true,
        ));
        id
    }

    pub fn log(&mut self, x: TensorId) -> TensorId {
        let out = B::log(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data.data,
            Arc::new(ops::Log),
            out.data.shape,
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
            out.data.data,
            Arc::new(ops::Mul),
            out.data.shape,
            vec![a, b],
            true,
        ));
        id
    }

    pub fn mean(&mut self, x: TensorId) -> TensorId {
        let out = B::mean(&self.nodes[x]);
        let id = self.nodes.len();

        self.nodes.push(Tensor::new(
            id,
            out.data.data,
            Arc::new(ops::Mean),
            out.data.shape,
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
            out.data.data,
            Arc::new(ops::MatMul),
            out.data.shape,
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
            out.data.data,
            Arc::new(ops::Sigmoid),
            out.data.shape,
            vec![x],
            true,
        ));
        id
    }

    pub fn backtrack(&mut self, loss: TensorId) {
        for n in &mut self.nodes {
            let grad_data = vec![T::zero(); n.data.data.len()];
            n.grad = NdimVector {
                data: grad_data,
                shape: n.data.shape.clone(),
                stride: n.data.stride.clone(),
            }
        }

        // seed dL/dL = 1
        self.nodes[loss].grad.data[0] = T::one();

        let topo = self.topo_from(loss);

        // reverse topo: loss → leaves
        for &id in topo.iter().rev() {
            self.backward_node(id);
        }
    }

    fn backward_node(&mut self, id: TensorId) {
        let (op, parents, grad, _out_data) = {
            let n = &self.nodes[id];
            (
                n.op.clone(),
                n.parents.clone(),
                n.grad.clone(),
                n.data.clone(),
            )
        };
        let parent_nodes: Vec<&Tensor<T, B>> = parents.iter().map(|&p| &self.nodes[p]).collect();
        let grads = op.backward(
            &parent_nodes,
            &Tensor::new(0, grad.data, Arc::new(ops::NoOp), grad.shape, vec![], false),
        );

        for (i, &p) in parents.iter().enumerate() {
            for (j, g) in self.nodes[p].grad.data.iter_mut().enumerate() {
                *g = *g + grads[i].data.data[j];
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for n in &mut self.nodes {
            for g in &mut n.grad.data {
                *g = T::zero();
            }
        }
    }

    pub fn step(&mut self, lr: f32) {
        for n in &mut self.nodes {
            if n.req_grad {
                for i in 0..n.data.data.len() {
                    let learning_rate = T::from_f32(lr).unwrap();
                    n.data.data[i] = n.data.data[i] - learning_rate * n.grad.data[i];
                }
            }
        }
    }

    fn topo_from(&self, start: TensorId) -> Vec<TensorId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = Vec::new();

        fn dfs<T: Numeric, B: Backend<DType = T>>(
            g: &Graph<T, B>,
            v: TensorId,
            visited: &mut Vec<bool>,
            stack: &mut Vec<TensorId>,
        ) {
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

    pub fn mul_scaler(&mut self, a: TensorId, _x: T) -> TensorId {
        let _node = &self.nodes[a];
        todo!()
    }
}
