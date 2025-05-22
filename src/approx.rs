use bon::Builder;
use rand::rngs::ThreadRng;
use rand::{thread_rng, RngCore};

use crate::util::{
    die_map, Entry, Value, APPROX_ACCURACY, APPROX_MAX_SAMPLE_SIZE, APPROX_MIN_SAMPLE_SIZE,
};
use crate::Die;

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
    pub fn approximate<K, F>(&mut self, mut op: F) -> Result<Die<K>, K::Error>
    where
        K: Clone + Copy + Ord + TryInto<f64>,
        F: FnMut(&mut G) -> K,
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

            s += k.try_into()?
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
            s += k.try_into()?;
            let mp = sp / f64::from(i - 1);
            let mc = s / f64::from(i);
            if (mp - mc).abs() < self.accuracy {
                denom = Some(i);
                break;
            }
        }

        let denom = Value::from(denom.unwrap_or(self.max_sample_size));
        println!("approx iterations: {denom}");

        Ok(Die::<K>::from_map(denom, outcomes))
    }
}
