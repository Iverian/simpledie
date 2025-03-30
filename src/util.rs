use std::collections::BTreeMap;

pub(crate) const SAMPLE_SIZE: u32 = 10_000_000;

pub(crate) type Count = u64;
pub(crate) type DieMap<K, V> = BTreeMap<K, V>;
