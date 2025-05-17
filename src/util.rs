use std::collections::BTreeMap;

pub(crate) const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
pub(crate) const APPROX_MIN_SAMPLE_SIZE: u32 = u16::MAX as u32;
pub(crate) const APPROX_ACCURACY: f64 = 1e-6;

pub(crate) type Key = i32;
pub(crate) type Value = u64;
pub(crate) type Probability = f64;
pub(crate) type DieMap = BTreeMap<Key, Value>;
pub(crate) type BigRatio = num::BigRational;
pub(crate) type BigInt = num::BigInt;
pub(crate) type BigUint = num::BigUint;
pub(crate) type Rc<T> = std::rc::Rc<T>;
