use std::collections::btree_map::Entry;

use rand::RngCore;

use crate::die::Die;
use crate::util::{DieMap, Key, APPROX_ACCURACY, APPROX_MAX_SAMPLE_SIZE, APPROX_MIN_SAMPLE_SIZE};

pub struct Approx<G>
where
    G: RngCore,
{
    rng: G,
}

impl<G> Approx<G>
where
    G: RngCore,
{
    #[must_use]
    pub fn new(rng: G) -> Self {
        Self { rng }
    }

    #[must_use]
    pub fn build<F>(&mut self, mut op: F) -> Die
    where
        F: FnMut(&mut G) -> Key,
    {
        let mut outcomes = DieMap::new();
        let mut s = 0f64;
        let mut denom = None;

        for _ in 0..APPROX_MIN_SAMPLE_SIZE {
            let k = op(&mut self.rng);

            match outcomes.entry(k) {
                Entry::Vacant(e) => {
                    e.insert(1);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }

            s += f64::from(k);
        }

        for i in APPROX_MIN_SAMPLE_SIZE..APPROX_MAX_SAMPLE_SIZE {
            let k = op(&mut self.rng);

            match outcomes.entry(k) {
                Entry::Vacant(e) => {
                    e.insert(1);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }

            let sp = s;
            s += f64::from(k);
            let mp = sp / f64::from(i - 1);
            let mc = s / f64::from(i);
            if (mp - mc).abs() < APPROX_ACCURACY {
                denom = Some(i);
                break;
            }
        }

        Die::from_map(denom.unwrap_or(APPROX_MAX_SAMPLE_SIZE), outcomes)
    }

    #[must_use]
    pub fn throws<P, F>(&mut self, die: Die, init: Key, pred: P, op: F) -> Die
    where
        P: Fn(Key) -> bool,
        F: Fn(Key, Key) -> Key,
    {
        let steps = Key::try_from(APPROX_MAX_SAMPLE_SIZE).unwrap_or(Key::MAX);
        self.build(move |g| {
            let mut value = init;
            for i in 0..steps {
                value = op(value, die.sample(g));
                if !pred(value) {
                    return i;
                }
            }
            steps
        })
    }
}
