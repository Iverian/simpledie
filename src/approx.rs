use bon::Builder;
use rand::rngs::ThreadRng;
use rand::{thread_rng, RngCore};

use crate::util::{die_map, Entry, APPROX_SAMPLE_SIZE};
use crate::{Die, Key};

#[derive(Debug, Builder)]
pub struct Approx<G = ThreadRng>
where
    G: RngCore,
{
    #[builder(finish_fn)]
    rng: G,
    #[builder(default = APPROX_SAMPLE_SIZE)]
    sample_size: u64,
}

impl Default for Approx<ThreadRng> {
    fn default() -> Self {
        Self::builder().build(thread_rng())
    }
}

impl<G> Approx<G>
where
    G: RngCore,
{
    #[must_use]
    pub fn approximate<K, F>(&mut self, mut op: F) -> Die<K>
    where
        K: Key,
        F: FnMut(&mut G) -> K,
    {
        let mut outcomes = die_map();
        for _ in 0..self.sample_size {
            let k = op(&mut self.rng);

            match outcomes.entry(k) {
                Entry::Vacant(e) => {
                    e.insert(1);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }
        }
        Die::<K>::from_map(self.sample_size, outcomes)
    }
}
