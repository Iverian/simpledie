use std::borrow::Borrow;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

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
pub struct Die<K> {
    denom: Count,
    keys: Vec<K>,
    outcomes: Vec<Count>,
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

impl<K> Die<K> {
    pub fn approx<G>(rng: &mut G) -> Approximator<'_, G>
    where
        G: RngCore,
    {
        Approximator::new(rng)
    }

    pub fn new(values: Vec<(K, u64)>) -> Self {
        let mut denom = BigUint::ZERO;
        let mut keys = Vec::with_capacity(values.len());
        let mut outcomes = Vec::with_capacity(values.len());
        for (k, c) in values {
            if c == 0 {
                continue;
            }
            let c = c.to_biguint().unwrap();
            denom += &c;
            keys.push(k);
            outcomes.push(c);
        }
        Self {
            denom,
            keys,
            outcomes,
        }
    }

    pub fn uniform_values(keys: Vec<K>) -> Self {
        let c = Count::one();
        let n = keys.len();
        Self {
            denom: n.to_biguint().unwrap(),
            keys,
            outcomes: vec![c; n],
        }
    }

    pub fn single(value: K) -> Self {
        Self {
            denom: Count::one(),
            keys: vec![value],
            outcomes: vec![Count::one()],
        }
    }

    pub fn cast<T>(self) -> Die<T>
    where
        T: From<K>,
    {
        self.biect_map(Into::into)
    }

    pub fn sample<G>(&self, rng: &mut G) -> &K
    where
        G: RngCore,
    {
        let v = rng.gen_biguint_range(&BigUint::ZERO, &self.denom);
        let mut pos = BigUint::ZERO;
        for (k, c) in self.zip() {
            pos += c;
            if &v < c {
                return k;
            }
        }
        unreachable!()
    }

    pub fn probabilities<T>(self) -> Option<Vec<(T, f64)>>
    where
        T: From<K>,
    {
        Self::into_zip(self.keys, self.outcomes)
            .map(|(k, c)| {
                Ratio::new(c, self.denom.clone())
                    .to_f64()
                    .map(|x| (k.into(), x))
            })
            .collect()
    }

    pub fn map<F, U>(self, op: F) -> Die<U>
    where
        F: Fn(K) -> U,
        U: Ord,
    {
        let mut outcomes = DieMap::<U, _>::new();
        for (k, c) in Self::into_zip(self.keys, self.outcomes) {
            match outcomes.entry(op(k)) {
                Entry::Vacant(e) => {
                    e.insert(c);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += c;
                }
            }
        }
        Die::<U>::from_map(self.denom, outcomes)
    }

    pub fn combine<F, U, Q, V>(dice: V, op: F) -> Die<U>
    where
        F: Fn(&[&K]) -> U,
        U: Ord,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        let dice = dice.borrow();
        let mut outcomes = DieMap::<U, _>::new();
        let mut key = Vec::with_capacity(dice.len());
        for p in dice
            .iter()
            .map(|x| x.borrow().zip())
            .multi_cartesian_product()
        {
            let mut count = Count::one();
            key.clear();
            for (k, c) in p {
                key.push(k);
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
            dice.iter()
                .fold(Count::one(), |acc, x| acc * &x.borrow().denom),
            outcomes,
        )
    }

    pub fn combine_with<F, T, U, Q>(&self, other: Q, op: F) -> Die<U>
    where
        F: Fn(&K, &T) -> U,
        U: Ord,
        Q: Borrow<Die<T>>,
    {
        let other = other.borrow();
        let mut outcomes = DieMap::<U, _>::new();
        for (k1, c1) in self.zip() {
            for (k2, c2) in other.zip() {
                match outcomes.entry(op(k1, k2)) {
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

    pub(crate) fn biect_map<F, U>(self, op: F) -> Die<U>
    where
        F: Fn(K) -> U,
    {
        Die::<U> {
            denom: self.denom,
            keys: self.keys.into_iter().map(op).collect(),
            outcomes: self.outcomes,
        }
    }

    fn from_map(denom: Count, value: DieMap<K, Count>) -> Self {
        let mut keys = Vec::with_capacity(value.len());
        let mut outcomes = Vec::with_capacity(value.len());
        for (k, v) in value {
            keys.push(k);
            outcomes.push(v);
        }
        Self {
            denom,
            keys,
            outcomes,
        }
    }

    fn into_zip(keys: Vec<K>, outcomes: Vec<Count>) -> impl Iterator<Item = (K, BigUint)> {
        keys.into_iter().zip(outcomes)
    }

    fn zip(&self) -> impl Iterator<Item = (&K, &BigUint)> + Clone {
        self.keys.iter().zip(self.outcomes.iter())
    }
}

impl<K> Die<K> where K: Copy {}

impl<K> Die<K>
where
    K: Ord + Clone,
{
    pub fn repeat<F>(&self, count: usize, op: F) -> Self
    where
        F: Fn(&K, &K) -> K + Copy,
    {
        let mut result = self.clone();
        if count < 2 {
            return result;
        }
        for _ in 1..count {
            result = result.combine_with(self, |x, y| op(x, y).clone());
        }
        result
    }

    pub fn mode(&self) -> Vec<&K> {
        self.zip()
            .max_set_by(|(_, x), (_, y)| x.cmp(y))
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }
}

impl<K> Die<K>
where
    K: TryInto<f64, Error: Clone + Debug> + Copy,
{
    pub fn mean(&self) -> Result<f64, K> {
        self.zip()
            .map(|(k, c)| Self::map_mean(*k, c.clone(), self.denom.clone()))
            .fold(Ok(0.0), Self::sum)
    }

    pub fn variance(&self) -> Result<f64, K> {
        let m = self.mean()?;
        self.zip()
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

impl Die<u32> {
    pub fn uniform(size: u32) -> Self {
        Self {
            denom: size.to_biguint().unwrap(),
            keys: (1..=size).collect(),
            outcomes: vec![Count::one(); size as usize],
        }
    }
}

impl<T> Die<T>
where
    T: Clone + Into<bool>,
{
    pub fn branch<K, U, V>(&self, lhs: U, rhs: V) -> Die<K>
    where
        K: Clone + Ord,
        U: Borrow<Die<K>>,
        V: Borrow<Die<K>>,
    {
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        let mut outcomes = DieMap::<K, _>::new();

        for (kc, cc) in self.zip() {
            let kc: bool = kc.clone().into();
            for (k1, c1) in lhs.zip() {
                for (k2, c2) in rhs.zip() {
                    let kk = if kc { k1.clone() } else { k2.clone() };
                    match outcomes.entry(kk) {
                        Entry::Vacant(e) => {
                            e.insert(cc * c1 * c2);
                        }
                        Entry::Occupied(mut e) => {
                            *e.get_mut() += cc * c1 * c2;
                        }
                    }
                }
            }
        }

        Die::<K>::from_map(&self.denom * &lhs.denom * &rhs.denom, outcomes)
    }
}

impl<K> Die<K>
where
    K: Clone + Ord,
{
    pub fn max(&self, rhs: &Self) -> Self {
        self.combine_with(rhs, |x, y| x.max(y).clone())
    }

    pub fn min(&self, rhs: &Self) -> Self {
        self.combine_with(rhs, |x, y| x.min(y).clone())
    }

    pub fn max_of(&self, count: usize) -> Self {
        self.repeat(count, |x, y| x.max(y).clone())
    }

    pub fn min_of(&self, count: usize) -> Self {
        self.repeat(count, |x, y| x.min(y).clone())
    }
}

impl<K> Die<K>
where
    K: Copy + Ord + Add<K, Output = K>,
{
    pub fn sum_of(&self, count: usize) -> Self {
        self.repeat(count, |x, y| *x + *y)
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
        Die {
            denom: denom.to_biguint().unwrap(),
            keys: outcomes.keys().copied().collect(),
            outcomes: outcomes
                .into_values()
                .map(|x| x.to_biguint().unwrap())
                .collect(),
        }
    }
}

impl<L, R> Add<Die<R>> for Die<L>
where
    L: Copy + Add<R>,
    R: Copy,
    <L as Add<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Add<R>>::Output>;

    fn add(self, rhs: Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x + *y)
    }
}

impl<L, R> Add<&Die<R>> for Die<L>
where
    L: Copy + Add<R>,
    R: Copy,
    <L as Add<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Add<R>>::Output>;

    fn add(self, rhs: &Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x + *y)
    }
}

impl<L, R> Add<Die<R>> for &Die<L>
where
    L: Copy + Add<R>,
    R: Copy,
    <L as Add<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Add<R>>::Output>;

    fn add(self, rhs: Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x + *y)
    }
}

impl<L, R> Add<&Die<R>> for &Die<L>
where
    L: Copy + Add<R>,
    R: Copy,
    <L as Add<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Add<R>>::Output>;

    fn add(self, rhs: &Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x + *y)
    }
}

impl<T, K> Add<T> for Die<K>
where
    K: Copy + Add<T>,
    T: Copy + Ord,
    <K as Add<T>>::Output: Copy,
{
    type Output = Die<<K as Add<T>>::Output>;

    fn add(self, rhs: T) -> Self::Output {
        self.biect_map(|x| x + rhs)
    }
}

impl<L, R> Sub<Die<R>> for Die<L>
where
    L: Copy + Sub<R>,
    R: Copy,
    <L as Sub<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Sub<R>>::Output>;

    fn sub(self, rhs: Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x - *y)
    }
}

impl<L, R> Sub<&Die<R>> for Die<L>
where
    L: Copy + Sub<R>,
    R: Copy,
    <L as Sub<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Sub<R>>::Output>;

    fn sub(self, rhs: &Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x - *y)
    }
}

impl<L, R> Sub<Die<R>> for &Die<L>
where
    L: Copy + Sub<R>,
    R: Copy,
    <L as Sub<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Sub<R>>::Output>;

    fn sub(self, rhs: Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x - *y)
    }
}

impl<L, R> Sub<&Die<R>> for &Die<L>
where
    L: Copy + Sub<R>,
    R: Copy,
    <L as Sub<R>>::Output: Ord + Copy,
{
    type Output = Die<<L as Sub<R>>::Output>;

    fn sub(self, rhs: &Die<R>) -> Self::Output {
        self.combine_with(rhs, |x, y| *x - *y)
    }
}

impl<T, K> Sub<T> for Die<K>
where
    K: Copy + Sub<T>,
    T: Copy + Ord,
    <K as Sub<T>>::Output: Copy,
{
    type Output = Die<<K as Sub<T>>::Output>;

    fn sub(self, rhs: T) -> Self::Output {
        self.biect_map(|x| x - rhs)
    }
}

impl<T, K> Mul<T> for Die<K>
where
    K: Copy + Ord + Add<K, Output = K>,
    T: Into<usize>,
{
    type Output = Die<K>;

    fn mul(self, rhs: T) -> Self::Output {
        self.repeat(rhs.into(), |x, y| *x + *y)
    }
}

impl<T, K> Mul<T> for &Die<K>
where
    K: Copy + Ord + Add<K, Output = K>,
    T: Into<usize>,
{
    type Output = Die<K>;

    fn mul(self, rhs: T) -> Self::Output {
        self.repeat(rhs.into(), |x, y| *x + *y)
    }
}
