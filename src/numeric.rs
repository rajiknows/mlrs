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
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn zero() -> Self;
}

impl Numeric for f64 {
    fn is_zero(&self) -> bool {
        if *self == f64::zero() { true } else { false }
    }
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
}
impl Numeric for f32 {
    fn is_zero(&self) -> bool {
        if *self == f32::zero() { true } else { false }
    }
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
}
impl Numeric for i32 {
    fn is_zero(&self) -> bool {
        if *self == i32::zero() { true } else { false }
    }
    fn one() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
}
impl Numeric for i64 {
    fn is_zero(&self) -> bool {
        if *self == i64::zero() { true } else { false }
    }
    fn one() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
}
