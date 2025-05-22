#![warn(clippy::pedantic)]
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

pub mod approx;
pub mod defs;
pub mod die;
pub mod expr;
pub mod prelude;
mod util;

use util::{Key, Value};

#[derive(Debug, Clone)]
pub struct Die<K = Key>
where
    K: Clone + Copy + Ord,
{
    denom: Value,
    keys: Vec<K>,
    outcomes: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Iter<'a, K>
where
    K: Clone + Copy + Ord,
{
    die: &'a Die<K>,
    index: usize,
}
