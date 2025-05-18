use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::die::Die;

pub const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
pub const APPROX_MIN_SAMPLE_SIZE: u32 = 50_000_000;
pub const APPROX_ACCURACY: f64 = 1e-9;
pub const DIRECT_MAX_DENOM: Value = 100_000_000;

pub type Key = i32;
pub type Value = u64;
pub type DieMap = BTreeMap<Key, Value>;
pub type Entry<'a> = std::collections::btree_map::Entry<'a, Key, Value>;
pub type DieList = Vec<Die>;
pub type BigRatio = num::BigRational;
pub type BigInt = num::BigInt;
pub type BigUint = num::BigUint;
pub type Cell<T> = Rc<RefCell<T>>;

#[inline]
pub fn die_map() -> DieMap {
    DieMap::new()
}

pub fn cell<T>(value: T) -> Cell<T> {
    Rc::new(RefCell::new(value))
}
