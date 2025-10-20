use std::collections::BTreeMap;

use thiserror::Error;

use crate::Die;

pub type Key = i32;
pub type Value = u64;
pub type DieMap = BTreeMap<Key, Value>;
pub type Entry<'a> = std::collections::btree_map::Entry<'a, Key, Value>;
pub type DieList = Vec<Die>;
pub type BigRatio = num::BigRational;
pub type BigInt = num::BigInt;
pub type BigUint = num::BigUint;

#[derive(Debug, Clone, Error)]
#[error("overflow in probabilities")]
pub struct OverflowError;

pub type OverflowResult<T> = Result<T, OverflowError>;

#[inline]
pub fn die_map() -> DieMap {
    DieMap::new()
}
