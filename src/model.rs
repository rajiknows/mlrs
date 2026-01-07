use crate::matrix::Matrix;

pub enum ModelVariableOps {
    MVOpsNull = 0,
    MVOpsCreate,
}

pub enum ModelVariableFlags {
    MVFlagNone = 0,
}

pub struct ModelVariable {
    pub flags: u32,
    pub index: u32,
    pub val: Matrix,
    pub grad: Matrix,
}
