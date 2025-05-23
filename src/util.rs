use std::collections::BTreeMap;

use thiserror::Error;

use crate::Die;

pub const APPROX_SAMPLE_SIZE: u64 = u32::MAX as u64;
pub const DIRECT_MAX_ITERATIONS: usize = 100_000_000;

pub type DefaultKey = i32;
pub type SignedValue = i64;
pub type Value = u64;
pub type DieMap<K> = BTreeMap<K, Value>;
pub type Entry<'a, K> = std::collections::btree_map::Entry<'a, K, Value>;
pub type DieList<K> = Vec<Die<K>>;
pub type BigRatio = num::BigRational;
pub type BigInt = num::BigInt;
pub type BigUint = num::BigUint;

#[derive(Debug, Clone, Error)]
#[error("overflow in probabilities")]
pub struct OverflowError;

pub type OverflowResult<T> = Result<T, OverflowError>;

#[inline]
pub fn die_map<T>() -> DieMap<T> {
    DieMap::new()
}
