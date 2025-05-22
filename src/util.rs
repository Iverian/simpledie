use std::collections::BTreeMap;

use thiserror::Error;

use crate::Die;

pub const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
pub const APPROX_MIN_SAMPLE_SIZE: u32 = 50_000_000;
pub const APPROX_ACCURACY: f64 = 1e-9;
pub const DIRECT_MAX_ITERATIONS: usize = 100_000_000;

pub type Key = i32;
pub type Value = u64;
pub type DieMap<K> = BTreeMap<K, Value>;
pub type Entry<'a, K> = std::collections::btree_map::Entry<'a, K, Value>;
pub type DieList<K> = Vec<Die<K>>;
pub type BigRatio = num::BigRational;
pub type BigInt = num::BigInt;
pub type BigUint = num::BigUint;

#[derive(Debug, Clone, Copy, Error)]
#[error("overflow in probabilities")]
pub struct OverflowError;

pub type OverflowResult<T> = ::core::result::Result<T, OverflowError>;

#[inline]
pub fn die_map<K>() -> DieMap<K>
where
    K: Clone + Copy + Ord,
{
    DieMap::new()
}
