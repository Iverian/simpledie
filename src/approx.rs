use std::collections::btree_map::Entry;

use rand::RngCore;

use crate::die::Die;
use crate::util::{DieMap, SAMPLE_SIZE};

pub struct Approx<G>
where
    G: RngCore,
{
    sample_size: u32,
    rng: G,
}

impl<G> Approx<G>
where
    G: RngCore,
{
    #[must_use]
    pub fn new(rng: G) -> Self {
        Self {
            sample_size: SAMPLE_SIZE,
            rng,
        }
    }

    pub fn sample_size(&self) -> u32 {
        self.sample_size
    }

    pub fn set_sample_size(&mut self, value: u32) {
        assert_ne!(value, 0);
        self.sample_size = value;
    }

    #[must_use]
    pub fn build<K, F>(&mut self, mut op: F) -> Die<K>
    where
        F: FnMut(&mut G) -> K,
        K: Ord,
    {
        let mut outcomes = DieMap::new();
        for _ in 0..self.sample_size {
            match outcomes.entry(op(&mut self.rng)) {
                Entry::Vacant(e) => {
                    e.insert(1u32);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }
        }
        Die::from_map(self.sample_size, outcomes)
    }

    #[must_use]
    pub fn throws<K, P, F>(&mut self, die: Die<K>, init: K, pred: P, op: F) -> Die<u32>
    where
        K: Clone,
        P: Fn(&K) -> bool,
        F: Fn(&K, &K) -> K,
    {
        let ss = self.sample_size;
        self.build(move |g| {
            let mut value = init.clone();
            for i in 0..ss {
                value = op(&value, die.sample(g));
                if !pred(&value) {
                    return i;
                }
            }
            ss
        })
    }
}
