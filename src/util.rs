use std::collections::BTreeMap;

pub(crate) const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
pub(crate) const APPROX_MIN_SAMPLE_SIZE: u32 = 10_000_000;
pub(crate) const APPROX_ACCURACY: f64 = 1e-6;
pub(crate) const DIRECT_MAX_DENOM: Value = 100_000_000;

#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_DENOM: Value = 1_000_000_000;
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_CHUNK_SIZE: usize = 10_000_000;

pub(crate) type Key = i32;
pub(crate) type Value = u128;
pub(crate) type Probability = f64;
pub(crate) type DieMap = BTreeMap<Key, Value>;
pub(crate) type Entry<'a> = std::collections::btree_map::Entry<'a, Key, Value>;
pub(crate) type BigRatio = num::BigRational;
pub(crate) type BigInt = num::BigInt;
pub(crate) type BigUint = num::BigUint;
pub(crate) type Rc<T> = std::rc::Rc<T>;
pub(crate) type FnPtr<T> = Box<T>;

#[inline]
pub(crate) fn die_map() -> DieMap {
    DieMap::new()
}
