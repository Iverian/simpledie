mod approx;
mod die;
mod expr;
pub mod ops;
mod value;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use approx::Approx;
pub use die::Die;
use thiserror::Error;
pub use value::{ComputableValue, DefaultValue, OrderedValue, Value};

type Outcome = u128;
type Map<T> = BTreeMap<T, Outcome>;
type Ptr<T> = Arc<T>;
type Result<T> = ::core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Error)]
#[error("overflow error")]
pub struct Error;

const DIRECT_MAX_ITERATIONS: usize = 100_000_000_000;
