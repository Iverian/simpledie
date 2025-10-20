use std::collections::BTreeMap;

use bon::Builder;
use rand::rngs::ThreadRng;
use rand::{rng, RngCore};

use crate::die::Outcome;
use crate::{
    ComputableValue, Die, APPROX_ACCURACY, APPROX_MAX_SAMPLE_SIZE, APPROX_MIN_SAMPLE_SIZE,
};

#[derive(Debug, Builder)]
pub struct Approx<G = ThreadRng>
where
    G: RngCore,
{
    #[builder(finish_fn)]
    rng: G,
    #[builder(default = APPROX_ACCURACY)]
    accuracy: f64,
    #[builder(default = APPROX_MIN_SAMPLE_SIZE)]
    min_sample_size: u32,
    #[builder(default = APPROX_MAX_SAMPLE_SIZE)]
    max_sample_size: u32,
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
    pub fn eval<T, F>(&mut self, mut op: F) -> Die<T>
    where
        T: ComputableValue,
        F: FnMut(&mut G) -> T,
    {
        let mut outcomes: BTreeMap<T, u128> = BTreeMap::new();
        let mut s = 0f64;
        let mut denom = None;

        for _ in 1..self.min_sample_size {
            let k = op(&mut self.rng);
            s += k.compute_f64();
            let e = outcomes.entry(k).or_default();
            *e += 1;
        }

        for i in self.min_sample_size..self.max_sample_size {
            let k = op(&mut self.rng);
            let sp = s;
            s += k.compute_f64();
            let e = outcomes.entry(k).or_default();
            *e += 1;

            let mp = sp / f64::from(i - 1);
            let mc = s / f64::from(i);
            if (mp - mc).abs() < self.accuracy {
                denom = Some(i as Outcome);
                break;
            }
        }

        Die::from_map(outcomes, denom.unwrap_or(self.max_sample_size as Outcome))
    }
}
