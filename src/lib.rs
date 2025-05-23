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
pub struct Die {
    denom: Value,
    keys: Vec<Key>,
    outcomes: Vec<Value>,
}

#[derive(Clone, Copy, Debug)]
pub enum EvalStrategy {
    Any,
    Approximate,
    Exact,
}
