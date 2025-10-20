mod approx;
pub mod defs;
mod die;
mod duality;
mod expr;
pub mod prelude;
mod value;

pub use approx::Approx;
pub use die::Die;
pub use value::{ComputableValue, OrderedValue, Value};

const APPROX_MAX_SAMPLE_SIZE: u32 = u32::MAX;
const APPROX_MIN_SAMPLE_SIZE: u32 = 50_000_000;
const APPROX_ACCURACY: f64 = 1e-9;
