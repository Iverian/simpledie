#![warn(clippy::pedantic)]
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

pub mod approx;
pub mod defs;
pub mod die;
mod expr;
pub mod prelude;
mod util;

use std::fmt::Debug;

pub use expr::composite::Composite;
pub use expr::ext::Expr;
pub use expr::Operation;
use util::{DefaultKey, Value};

pub trait Key: Copy + Ord + Debug {}

#[derive(Debug, Clone)]
pub struct Die<T = DefaultKey>
where
    T: Key,
{
    denom: Value,
    keys: Vec<T>,
    outcomes: Vec<Value>,
}

#[derive(Clone, Copy, Debug)]
pub enum EvalStrategy {
    Any,
    Approximate,
    Exact,
}

impl<T> Key for T where T: Copy + Ord + Debug {}
