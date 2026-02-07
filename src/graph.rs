use std::sync::Arc;

use crate::{
    backends::Backend,
    numeric::Numeric,
    ops,
    tensor::{Tensor, TensorId},
    utils::calculate_strides,
};

/// The Graph only deals with creating the computational graph
/// and does not care about the implementation of functions from backend
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

        let inner = B::add(&node_a.inner, &node_b.inner);

        let id = self.nodes.len();
        let stride = calculate_strides(&node_a.shape);

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

        assert_eq!(node_a.shape, node_b.shape);

        let inner = B::sub(&node_a.inner, &node_b.inner);

        let id = self.nodes.len();
        let stride = calculate_strides(&node_a.shape);

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node_a.shape.clone(),
            stride,
            op: Arc::new(ops::Sub),
            parents: vec![a, b],
            req_grad: node_a.req_grad || node_b.req_grad,
        });

        id
    }

    // broadcast b to a
    pub fn add_broadcast(&mut self, into_tensor: TensorId, broadcast_tensor: TensorId) -> TensorId {
        let node_a = &self.nodes[into_tensor];
        let node_b = &self.nodes[broadcast_tensor];

        if node_a.shape == node_b.shape {
            return self.add(into_tensor, broadcast_tensor);
        }

        let b = B::broadcast(&node_b.inner, &node_a.shape);
        let inner = B::add(&node_a.inner, &b);

        let id = self.nodes.len();
        let stride = calculate_strides(&node_a.shape);

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node_a.shape.clone(),
            stride,
            op: Arc::new(ops::AddBroadcast),
            parents: vec![into_tensor, broadcast_tensor],
            req_grad: node_a.req_grad || node_b.req_grad,
        });

        id
    }

    pub fn neg(&mut self, x: TensorId) -> TensorId {
        let node = &self.nodes[x];
        let inner = B::neg(&node.inner);
        let id = self.nodes.len();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node.shape.clone(),
            stride: calculate_strides(&node.shape),
            op: Arc::new(ops::Neg),
            parents: vec![x],
            req_grad: node.req_grad,
        });
        id
    }

    pub fn log(&mut self, x: TensorId) -> TensorId {
        let node = &self.nodes[x];
        let inner = B::log(&node.inner);
        let id = self.nodes.len();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node.shape.clone(),
            stride: calculate_strides(&node.shape),
            op: Arc::new(ops::Log),
            parents: vec![x],
            req_grad: node.req_grad,
        });
        id
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        assert_eq!(node_a.shape, node_b.shape);

        let inner = B::mul(&node_a.inner, &node_b.inner);
        let id = self.nodes.len();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node_a.shape.clone(),
            stride: calculate_strides(&node_a.shape),
            op: Arc::new(ops::Mul),
            parents: vec![a, b],
            req_grad: node_a.req_grad || node_b.req_grad,
        });
        id
    }

    pub fn mean(&mut self, x: TensorId) -> TensorId {
        let node = &self.nodes[x];
        let inner = B::mean(&node.inner);
        let id = self.nodes.len();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: vec![1],
            stride: calculate_strides(&[1]),
            op: Arc::new(ops::Mean),
            parents: vec![x],
            req_grad: node.req_grad,
        });
        id
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];
        let inner = B::matmul(&node_a.inner, &node_b.inner, &node_a.shape, &node_b.shape);
        let id = self.nodes.len();

        let out_shape = vec![node_a.shape[0], node_b.shape[1]];
        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: out_shape.clone(),
            stride: calculate_strides(&out_shape),
            op: Arc::new(ops::MatMul),
            parents: vec![a, b],
            req_grad: node_a.req_grad || node_b.req_grad,
        });
        id
    }

    pub fn sigmoid(&mut self, x: TensorId) -> TensorId {
        let node = &self.nodes[x];
        let inner = B::sigmoid(&node.inner);
        let id = self.nodes.len();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: node.shape.clone(),
            stride: calculate_strides(&node.shape),
            op: Arc::new(ops::Sigmoid),
            parents: vec![x],
            req_grad: node.req_grad,
        });
        id
    }

    pub fn backtrack(&mut self, loss: TensorId) {
        for n in &mut self.nodes {
            n.grad = Some(B::fill(&n.inner, T::zero()));
        }

        let loss_shape = self.nodes[loss].shape.clone();
        let numel = loss_shape.iter().product::<usize>().max(1);
        let mut grad_data = vec![T::zero(); numel];
        if !grad_data.is_empty() {
            grad_data[0] = T::one();
        }
        self.nodes[loss].grad = Some(B::from_cpu(&grad_data, &loss_shape));

        let topo = self.topo_from(loss);

        // reverse topo: loss → leaves
        for &id in topo.iter().rev() {
            self.backward_node(id);
        }
    }

    fn backward_node(&mut self, id: TensorId) {
        let (op, parents) = {
            let n = &self.nodes[id];
            (n.op.clone(), n.parents.clone())
        };

        if parents.is_empty() {
            return;
        }

        let grad = match self.nodes[id].grad.as_ref() {
            Some(g) => g,
            None => return,
        };

        let parent_nodes: Vec<&Tensor<T, B>> = parents.iter().map(|&p| &self.nodes[p]).collect();
        let grads = op.backward(&parent_nodes, grad);

        for (p, g) in parents.into_iter().zip(grads.into_iter()) {
            let existing = self.nodes[p].grad.take();
            let new_grad = match existing {
                Some(existing) => B::add(&existing, &g),
                None => g,
            };
            self.nodes[p].grad = Some(new_grad);
        }
    }

    pub fn zero_grad(&mut self) {
        for n in &mut self.nodes {
            n.grad = Some(B::fill(&n.inner, T::zero()));
        }
    }

    pub fn step(&mut self, lr: f32) {
        let lr_t = T::from_f32(lr).unwrap();
        for n in &mut self.nodes {
            if n.req_grad != true {
                continue;
            }
            if let Some(grad) = n.grad.as_ref() {
                let lr_tensor = B::fill(grad, lr_t);
                let scaled = B::mul(&lr_tensor, grad);
                n.inner = B::sub(&n.inner, &scaled);
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

    pub fn mul_scaler(&mut self, a: TensorId, x: T) -> TensorId {
        let node = &self.nodes[a];
        let inner = B::fill(&node.inner, x);
        let id = self.nodes.len();
        let shape = node.shape.clone();

        self.nodes.push(Tensor {
            id,
            inner,
            grad: None,
            shape: shape.clone(),
            stride: calculate_strides(&shape),
            op: Arc::new(ops::NoOp),
            parents: Vec::new(),
            req_grad: false,
        });

        self.mul(a, id)
    }
}
