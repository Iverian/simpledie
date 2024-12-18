use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt::Debug;

use itertools::Itertools;
use num::bigint::{RandBigInt, ToBigUint};
use num::rational::Ratio;
use num::traits::One;
use num::{BigUint, ToPrimitive};
use rand::RngCore;
use thiserror::Error;

const SAMPLE_SIZE: usize = 10_000_000;

type Count = BigUint;
type DieMap<K, V> = BTreeMap<K, V>;

#[derive(Debug, Clone)]
pub struct Die<K>
where
    K: Copy,
{
    outcomes: Vec<(K, Count)>,
    denom: Count,
}

pub struct Approximator<'a, G>
where
    G: RngCore,
{
    sample_size: usize,
    rng: &'a mut G,
}

#[derive(Debug, Clone, Error)]
pub enum Error<K>
where
    K: TryInto<f64>,
    <K as TryInto<f64>>::Error: Clone + Debug,
{
    #[error("float conversion failed: {0:?}")]
    Float(<K as TryInto<f64>>::Error),
    #[error("overflow in probability")]
    Overflow,
}

pub type Result<T, K> = core::result::Result<T, Error<K>>;

impl<K> Die<K>
where
    K: Copy,
{
    pub fn approx<G: RngCore>(rng: &mut G) -> Approximator<'_, G> {
        Approximator::new(rng)
    }

    pub fn single(value: K) -> Self {
        Self::new(Count::one(), vec![(value, Count::one())])
    }

    pub fn cast<T>(self) -> Die<T>
    where
        T: From<K> + Copy,
    {
        Die::<T>::new(
            self.denom,
            self.outcomes
                .into_iter()
                .map(|(k, c)| (k.into(), c))
                .collect(),
        )
    }

    pub fn map<F, U>(self, op: F) -> Die<U>
    where
        F: Fn(K) -> U,
        U: Ord + Copy,
    {
        let mut outcomes = DieMap::<U, _>::new();
        for (k, c) in self.outcomes {
            match outcomes.entry(op(k)) {
                Entry::Vacant(e) => {
                    e.insert(c);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += c;
                }
            }
        }
        Die::<U>::from_map(self.denom.clone(), outcomes)
    }

    pub fn combine<F, U>(dice: &[&Self], op: F) -> Die<U>
    where
        F: Fn(&[K]) -> U,
        U: Ord + Copy,
    {
        let mut outcomes = DieMap::<U, _>::new();
        let mut key = Vec::with_capacity(dice.len());
        for p in dice.iter().map(|x| &x.outcomes).multi_cartesian_product() {
            let mut count = Count::one();
            key.clear();
            for (k, c) in p {
                key.push(*k);
                count *= c;
            }
            match outcomes.entry(op(&key)) {
                Entry::Vacant(e) => {
                    e.insert(count);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += count;
                }
            }
        }
        Die::<U>::from_map(
            dice.iter().fold(Count::one(), |acc, x| acc * &x.denom),
            outcomes,
        )
    }

    pub fn combine_with<F, T, U>(&self, other: &Die<T>, op: F) -> Die<U>
    where
        F: Fn(K, T) -> U,
        T: Copy,
        U: Ord + Copy,
    {
        let mut outcomes = DieMap::<U, _>::new();
        for (k1, c1) in &self.outcomes {
            for (k2, c2) in &other.outcomes {
                match outcomes.entry(op(*k1, *k2)) {
                    Entry::Vacant(e) => {
                        e.insert(c1 * c2);
                    }
                    Entry::Occupied(mut e) => {
                        *e.get_mut() += c1 * c2;
                    }
                }
            }
        }
        Die::<U>::from_map(&self.denom * &other.denom, outcomes)
    }

    pub fn probabilities<T>(&self) -> Option<Vec<(T, f64)>>
    where
        T: From<K>,
    {
        self.outcomes
            .iter()
            .map(|(k, c)| {
                Ratio::new(c.clone(), self.denom.clone())
                    .to_f64()
                    .map(|x| ((*k).into(), x))
            })
            .collect()
    }

    pub fn sample<G>(&self, rng: &mut G) -> K
    where
        G: RngCore,
    {
        let v = rng.gen_biguint_range(&BigUint::ZERO, &self.denom);
        let mut pos = BigUint::ZERO;
        for (k, c) in &self.outcomes {
            pos += c;
            if &v < c {
                return *k;
            }
        }
        unreachable!()
    }

    fn from_map(denom: Count, value: DieMap<K, Count>) -> Self {
        Self::new(denom, value.into_iter().collect())
    }

    fn new(denom: Count, outcomes: Vec<(K, Count)>) -> Self {
        Self { outcomes, denom }
    }
}

impl<K> Die<K>
where
    K: Ord + Copy,
{
    pub fn repeat<F>(&self, count: usize, op: F) -> Self
    where
        F: Fn(K, K) -> K + Copy,
    {
        let mut result = self.clone();
        if count < 2 {
            return result;
        }
        for _ in 1..count {
            result = result.combine_with(self, op);
        }
        result
    }
}

impl<K> Die<K>
where
    K: TryInto<f64, Error: Clone + Debug> + Copy,
{
    pub fn mean(&self) -> Result<f64, K> {
        self.outcomes
            .iter()
            .map(|(k, c)| Self::map_mean(*k, c.clone(), self.denom.clone()))
            .fold(Ok(0.0), Self::sum)
    }

    pub fn variance(&self) -> Result<f64, K> {
        let m = self.mean()?;
        self.outcomes
            .iter()
            .map(|(k, c)| Self::map_variance(m, *k, c.clone(), self.denom.clone()))
            .fold(Ok(0.0), Self::sum)
    }

    pub fn stddev(&self) -> Result<f64, K> {
        self.variance().map(|x| x.sqrt())
    }

    fn map_mean(k: K, c: Count, d: Count) -> Result<f64, K> {
        let kk: f64 = k.try_into().map_err(|e| Error::Float(e))?;
        let cc = Ratio::new_raw(c, d).to_f64().ok_or(Error::Overflow)?;
        Ok(kk * cc)
    }

    fn map_variance(m: f64, k: K, c: Count, d: Count) -> Result<f64, K> {
        let kk: f64 = k.try_into().map_err(|e| Error::Float(e))?;
        let cc = Ratio::new_raw(c, d).to_f64().ok_or(Error::Overflow)?;
        Ok((kk - m).powi(2) * cc)
    }

    fn sum(acc: Result<f64, K>, x: Result<f64, K>) -> Result<f64, K> {
        Ok(acc? + x?)
    }
}

impl<K> Die<K>
where
    K: Ord + Copy,
{
    pub fn median(&self) -> K {
        self.outcomes[(self.outcomes.len() - 1) / 2].0
    }

    pub fn mode(&self) -> Vec<K> {
        self.outcomes
            .iter()
            .max_set_by_key(|(_, c)| c)
            .into_iter()
            .map(|(k, _)| *k)
            .collect()
    }
}

impl Die<u32> {
    pub fn uniform(size: u32) -> Self {
        let c = Count::one();
        Self {
            outcomes: (1..=size).map(|x| (x, c.clone())).collect(),
            denom: size.to_biguint().unwrap(),
        }
    }
}

impl<'a, G> Approximator<'a, G>
where
    G: RngCore,
{
    pub fn new(rng: &'a mut G) -> Self {
        Self {
            sample_size: SAMPLE_SIZE,
            rng,
        }
    }

    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn set_sample_size(mut self, value: usize) -> Self {
        assert_ne!(value, 0);
        self.sample_size = value;
        self
    }

    pub fn build<K, F>(&mut self, mut op: F) -> Die<K>
    where
        F: FnMut(&mut G) -> K,
        K: Ord + Copy,
    {
        let mut outcomes = DieMap::new();
        for _ in 0..self.sample_size {
            match outcomes.entry(op(self.rng)) {
                Entry::Vacant(e) => {
                    e.insert(1u64);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }
        }
        Self::convert(outcomes, SAMPLE_SIZE)
    }

    fn convert<K>(outcomes: DieMap<K, u64>, denom: usize) -> Die<K>
    where
        K: Ord + Copy,
    {
        Die::new(
            denom.to_biguint().unwrap(),
            outcomes
                .into_iter()
                .map(|(k, v)| (k, v.to_biguint().unwrap()))
                .collect(),
        )
    }
}
