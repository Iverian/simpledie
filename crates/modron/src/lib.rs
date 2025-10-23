mod approx;
mod die;
mod expr;
pub mod ops;
mod value;

use std::sync::Arc;

use ahash::AHashMap;
pub use approx::Approx;
pub use die::Die;
use thiserror::Error;
pub use value::{ComputableValue, ComputedValue, DefaultValue, OrderedValue, Value};

type Outcome = u128;
type Map<T> = AHashMap<T, Outcome>;
type Ptr<T> = Arc<T>;
type Result<T> = ::core::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Error)]
#[error("overflow error")]
pub struct Error;

const DIRECT_MAX_ITERATIONS: usize = 100_000_000_000;
const MIN_EXPLODE: usize = 1;
const MAX_EXPLODE: usize = 10;
