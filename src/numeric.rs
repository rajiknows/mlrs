use serde::{Serialize, de::DeserializeOwned};
use std::ops;

pub trait Numeric:
    Copy
    + Clone
    + std::fmt::Debug
    + PartialEq
    + ops::Add<Output = Self>
    + ops::AddAssign
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + ops::Div<Output = Self>
    + ops::Neg<Output = Self>
    + PartialOrd
    + Default
    + Serialize
    + DeserializeOwned
{
}

impl Numeric for f64 {}
impl Numeric for f32 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
