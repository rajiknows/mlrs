// in this module we will define a model struct and the operation we will perform with a model
//
//
// the tensor struct and its op were kinda similar but i don't like the api that much so
// here we are
//

use crate::tensor::Tensor;

pub struct Node<'a> {
    // so what does a model have
    // it has input
    pub tensor: &'a Tensor,
    pub op_fn: &'a dyn FnOnce(),
}
