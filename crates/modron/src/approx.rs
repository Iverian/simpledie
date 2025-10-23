use std::sync::{LazyLock, RwLock};

use bon::Builder;
use rand::rngs::ThreadRng;
use rand::{rng, RngCore};

use crate::die::DieInner;
use crate::{Die, Map, Outcome, Value};

const APPROX_MIN_SAMPLE_SIZE: u32 = 50_000_000;

static APPROX_SAMPLE_SIZE: LazyLock<RwLock<u32>> =
    LazyLock::new(|| RwLock::new(APPROX_MIN_SAMPLE_SIZE));

#[derive(Debug, Builder)]
pub struct Approx<G = ThreadRng>
where
    G: RngCore,
{
    #[builder(finish_fn)]
    rng: G,
    #[builder(default = *APPROX_SAMPLE_SIZE.read().unwrap())]
    sample_size: u32,
}

impl<G> Approx<G>
where
    G: RngCore,
{
    pub fn set_default_sample_size(value: u32) {
        let value = value.clamp(APPROX_MIN_SAMPLE_SIZE, u32::MAX);
        let mut guard = APPROX_SAMPLE_SIZE.write().unwrap();
        *guard = value;
    }
}

impl Default for Approx<ThreadRng> {
    fn default() -> Self {
        Self::builder().build(rng())
    }
}

impl<G> Approx<G>
where
    G: RngCore,
{
    #[must_use]
    pub fn eval<T, F>(&mut self, op: F) -> Die<T>
    where
        T: Value,
        F: FnMut(&mut G) -> T,
    {
        Die::new(self.eval_inner(op))
    }

    #[must_use]
    pub(crate) fn eval_inner<T, F>(&mut self, mut op: F) -> DieInner<T>
    where
        T: Value,
        F: FnMut(&mut G) -> T,
    {
        let mut map = Map::new();

        for _ in 0..self.sample_size {
            let k = op(&mut self.rng);
            let e = map.entry(k).or_default();
            *e += 1;
        }

        DieInner::new(map, self.sample_size as Outcome)
    }
}
