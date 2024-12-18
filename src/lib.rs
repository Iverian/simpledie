#![warn(clippy::pedantic)]
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
mod defs;
mod die;

pub use defs::*;
pub use die::{Approx, Die, Error, Result};
