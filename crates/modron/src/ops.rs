use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::Value;

pub fn add<L, R>(lhs: &L, rhs: &R) -> L::Output
where
    L: Value,
    R: Value,
    L: Add<R>,
    L::Output: Value,
{
    lhs.clone().add(rhs.clone())
}

pub fn sub<L, R>(lhs: &L, rhs: &R) -> L::Output
where
    L: Value,
    R: Value,
    L: Sub<R>,
    L::Output: Value,
{
    lhs.clone().sub(rhs.clone())
}

pub fn mul<L, R>(lhs: &L, rhs: &R) -> L::Output
where
    L: Value,
    R: Value,
    L: Mul<R>,
    L::Output: Value,
{
    lhs.clone().mul(rhs.clone())
}

pub fn div<L, R>(lhs: &L, rhs: &R) -> L::Output
where
    L: Value,
    R: Value,
    L: Div<R>,
    L::Output: Value,
{
    lhs.clone().div(rhs.clone())
}

pub fn rem<L, R>(lhs: &L, rhs: &R) -> L::Output
where
    L: Value,
    R: Value,
    L: Rem<R>,
    L::Output: Value,
{
    lhs.clone().rem(rhs.clone())
}
