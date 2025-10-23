mod inner;
mod ptr;

use std::iter::Zip;
use std::slice;

pub use inner::DieInner;
pub use ptr::Die;
use rand::RngCore;

use crate::value::Value;
use crate::Outcome;

pub type Iter<'a, T> = Zip<slice::Iter<'a, T>, slice::Iter<'a, Outcome>>;

pub trait DieLike<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome;
    fn values(&self) -> &[T];
    fn outcomes(&self) -> &[Outcome];
    fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore;
}
