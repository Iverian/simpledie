use bon::Builder;
use rand::rngs::ThreadRng;
use rand::{thread_rng, RngCore};

use crate::die::Die;
use crate::util::{
    die_map, Entry, Key, APPROX_ACCURACY, APPROX_MAX_SAMPLE_SIZE, APPROX_MIN_SAMPLE_SIZE,
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
        Self::builder().build(thread_rng())
    }
}

impl<G> Approx<G>
where
    G: RngCore,
{
    #[must_use]
    pub fn approximate<F>(&mut self, mut op: F) -> Die
    where
        F: FnMut(&mut G) -> Key,
    {
        let mut outcomes = die_map();
        let mut s = 0f64;
        let mut denom = None;

        for _ in 1..self.min_sample_size {
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

        for i in self.min_sample_size..self.max_sample_size {
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
            if (mp - mc).abs() < self.accuracy {
                denom = Some(i);
                break;
            }
        }

        Die::from_map(denom.unwrap_or(self.max_sample_size), outcomes)
    }

    #[must_use]
    pub fn count_throws<P, F>(&mut self, die: Die, init: Key, pred: P, op: F) -> Die
    where
        P: Fn(Key) -> bool,
        F: Fn(Key, Key) -> Key,
    {
        let steps = Key::try_from(self.max_sample_size).unwrap_or(Key::MAX);
        self.approximate(move |g| {
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
