use std::collections::BTreeMap;

use num::BigUint;

pub(crate) const SAMPLE_SIZE: u32 = 10_000_000;

pub(crate) type Count = BigUint;
pub(crate) type DieMap<K, V> = BTreeMap<K, V>;
