use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::die::Die;

pub(crate) const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
pub(crate) const APPROX_MIN_SAMPLE_SIZE: u32 = 50_000_000;
pub(crate) const APPROX_ACCURACY: f64 = 1e-9;
pub(crate) const DIRECT_MAX_DENOM: Value = 100_000_000;

pub(crate) type Key = i32;
pub(crate) type Value = u64;
pub(crate) type DieMap = BTreeMap<Key, Value>;
pub(crate) type Entry<'a> = std::collections::btree_map::Entry<'a, Key, Value>;
pub(crate) type DieList = Vec<Die>;
pub(crate) type BigRatio = num::BigRational;
pub(crate) type BigInt = num::BigInt;
pub(crate) type BigUint = num::BigUint;
pub(crate) type Cell<T> = Rc<RefCell<T>>;

#[inline]
pub(crate) fn die_map() -> DieMap {
    DieMap::new()
}

pub(crate) fn cell<T>(value: T) -> Cell<T> {
    Rc::new(RefCell::new(value))
}
