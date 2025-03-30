use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

use itertools::Itertools;
use num::rational::Ratio;
use num::traits::One;
use num::ToPrimitive;
use rand::{thread_rng, Rng, RngCore};
use thiserror::Error;

use crate::approx::Approx;
use crate::die_list::DieList;
use crate::util::{Count, DieMap, SAMPLE_SIZE};

#[derive(Debug, Clone)]
pub struct Die<K> {
    denom: Count,
    keys: Vec<K>,
    outcomes: Vec<Count>,
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
    #[must_use]
    pub fn new(values: Vec<(K, Count)>) -> Self {
        let mut denom = 0;
        let mut keys = Vec::with_capacity(values.len());
        let mut outcomes = Vec::with_capacity(values.len());
        for (k, c) in values {
            if c == 0 {
                continue;
            }
            denom += c;
            keys.push(k);
            outcomes.push(c);
        }
        Self {
            denom,
            keys,
            outcomes,
        }
    }

    #[must_use]
    pub(crate) fn from_map<T>(denom: T, value: DieMap<K, T>) -> Self
    where
        T: Into<Count>,
    {
        let mut keys = Vec::with_capacity(value.len());
        let mut outcomes = Vec::with_capacity(value.len());
        for (k, v) in value {
            keys.push(k);
            outcomes.push(v.into());
        }
        Self {
            denom: denom.into(),
            keys,
            outcomes,
        }
    }

    #[must_use]
    pub(crate) fn from_raw_map(denom: Count, value: DieMap<K, Count>) -> Self {
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

    #[must_use]
    pub fn uniform_values(keys: Vec<K>) -> Self {
        let c = Count::one();
        let n = keys.len();
        Self {
            denom: Count::try_from(n).unwrap(),
            keys,
            outcomes: vec![c; n],
        }
    }

    #[must_use]
    pub fn single(value: K) -> Self {
        Self {
            denom: Count::one(),
            keys: vec![value],
            outcomes: vec![Count::one()],
        }
    }

    #[must_use]
    pub fn cast<T>(self) -> Die<T>
    where
        T: From<K>,
    {
        self.biect_map(Into::into)
    }

    pub fn try_cast<T>(self) -> core::result::Result<Die<T>, <T as TryFrom<K>>::Error>
    where
        T: TryFrom<K>,
        <T as TryFrom<K>>::Error: Clone + Debug,
    {
        self.try_biect_map(TryInto::try_into)
    }

    #[must_use]
    pub fn sample<G>(&self, rng: &mut G) -> &K
    where
        G: RngCore,
    {
        let v = rng.gen_range(0u64..self.denom);
        let mut pos = 0;
        for (k, c) in self.zip() {
            pos += c;
            if v < pos {
                return k;
            }
        }
        unreachable!()
    }

    #[must_use]
    pub fn probabilities<T>(self) -> Option<Vec<(T, f64)>>
    where
        T: From<K>,
    {
        Self::into_zip(self.keys, self.outcomes)
            .map(|(k, c)| Ratio::new(c, self.denom).to_f64().map(|x| (k.into(), x)))
            .collect()
    }

    #[must_use]
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
        Die::<U>::from_raw_map(self.denom, outcomes)
    }

    #[must_use]
    pub fn combine<F, U, Q, V>(dice: V, op: F) -> Die<U>
    where
        F: Fn(&[&K]) -> U,
        U: Ord,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        let denom = dice
            .borrow()
            .iter()
            .fold(Count::one(), |acc, x| acc * x.borrow().denom);
        if denom < Count::from(SAMPLE_SIZE) {
            Self::combine_raw(denom, dice, op)
        } else {
            Self::combine_approx(dice, op)
        }
    }

    #[must_use]
    fn combine_approx<F, U, Q, V>(dice: V, op: F) -> Die<U>
    where
        F: Fn(&[&K]) -> U,
        U: Ord,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        let dice = dice.borrow();
        Approx::new(thread_rng()).build(|rng| {
            let x: Vec<_> = dice.iter().map(|x| x.borrow().sample(rng)).collect();
            op(x.as_slice())
        })
    }

    #[must_use]
    fn combine_raw<F, U, Q, V>(denom: Count, dice: V, op: F) -> Die<U>
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
        Die::<U>::from_raw_map(denom, outcomes)
    }

    #[must_use]
    pub fn combine_with<F, T, U, Q>(&self, other: Q, op: F) -> Die<U>
    where
        F: Fn(&K, &T) -> U,
        U: Ord,
        Q: Borrow<Die<T>>,
    {
        let denom = self.denom * other.borrow().denom;
        if denom < Count::from(SAMPLE_SIZE) {
            self.combine_with_raw(denom, other, op)
        } else {
            self.combine_with_approx(other, op)
        }
    }

    #[must_use]
    fn combine_with_approx<F, T, U, Q>(&self, other: Q, op: F) -> Die<U>
    where
        F: Fn(&K, &T) -> U,
        U: Ord,
        Q: Borrow<Die<T>>,
    {
        let other = other.borrow();
        Approx::new(thread_rng()).build(|rng| {
            let x = self.sample(rng);
            let y = other.sample(rng);
            op(x, y)
        })
    }

    #[must_use]
    fn combine_with_raw<F, T, U, Q>(&self, denom: Count, other: Q, op: F) -> Die<U>
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
        Die::<U>::from_raw_map(denom, outcomes)
    }

    pub(crate) fn try_biect_map<F, U, E>(self, op: F) -> core::result::Result<Die<U>, E>
    where
        F: Fn(K) -> core::result::Result<U, E>,
        E: Clone + Debug,
    {
        Ok(Die::<U> {
            denom: self.denom,
            keys: self
                .keys
                .into_iter()
                .map(op)
                .collect::<core::result::Result<Vec<_>, _>>()?,
            outcomes: self.outcomes,
        })
    }

    #[must_use]
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

    fn into_zip(keys: Vec<K>, outcomes: Vec<Count>) -> impl Iterator<Item = (K, Count)> {
        keys.into_iter().zip(outcomes)
    }

    fn zip(&self) -> impl Iterator<Item = (&K, &Count)> + Clone {
        self.keys.iter().zip(self.outcomes.iter())
    }
}

impl<K> Die<K>
where
    K: Clone,
{
    #[must_use]
    pub fn n(self, count: usize) -> DieList<K> {
        DieList::repeat(count, self)
    }
}

impl<K> Die<K>
where
    K: Ord + Clone,
{
    #[must_use]
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
            .map(|(k, c)| Self::map_mean(*k, *c, self.denom))
            .fold(Ok(0.0), Self::acc_sum)
    }

    pub fn variance(&self) -> Result<f64, K> {
        let m = self.mean()?;
        self.zip()
            .map(|(k, c)| Self::map_variance(m, *k, *c, self.denom))
            .fold(Ok(0.0), Self::acc_sum)
    }

    pub fn stddev(&self) -> Result<f64, K> {
        self.variance().map(f64::sqrt)
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

    fn acc_sum(acc: Result<f64, K>, x: Result<f64, K>) -> Result<f64, K> {
        Ok(acc? + x?)
    }
}

impl Die<i32> {
    #[must_use]
    pub fn uniform(size: u16) -> Self {
        Self {
            denom: Count::from(size),
            keys: (1..=i32::from(size)).collect(),
            outcomes: vec![Count::one(); size as usize],
        }
    }
}

impl<T> Die<T>
where
    T: Clone + Into<bool>,
{
    #[must_use]
    pub fn branch<K, U, V>(&self, lhs: U, rhs: V) -> Die<K>
    where
        K: Clone + Ord,
        U: Borrow<Die<K>>,
        V: Borrow<Die<K>>,
    {
        let denom = self.denom * lhs.borrow().denom * rhs.borrow().denom;
        if denom < Count::from(SAMPLE_SIZE) {
            self.branch_raw(denom, lhs, rhs)
        } else {
            self.branch_approx(lhs, rhs)
        }
    }

    #[must_use]
    fn branch_approx<K, U, V>(&self, lhs: U, rhs: V) -> Die<K>
    where
        K: Clone + Ord,
        U: Borrow<Die<K>>,
        V: Borrow<Die<K>>,
    {
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        Approx::new(thread_rng()).build(|rng| {
            let a = self.sample(rng).clone().into();
            if a { lhs.sample(rng) } else { rhs.sample(rng) }.clone()
        })
    }

    #[must_use]
    fn branch_raw<K, U, V>(&self, denom: Count, lhs: U, rhs: V) -> Die<K>
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
                    match outcomes.entry((if kc { k1 } else { k2 }).clone()) {
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

        Die::<K>::from_raw_map(denom, outcomes)
    }
}

impl<K> Die<K>
where
    K: Eq,
{
    #[must_use]
    pub fn eq<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x == value)
    }

    #[must_use]
    pub fn ne<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x != value)
    }
}

impl<K> Die<K>
where
    K: Ord,
{
    #[must_use]
    pub fn cmp<T>(self, value: T) -> Die<Ordering>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x.cmp(&value))
    }

    #[must_use]
    pub fn lt<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x < value)
    }

    #[must_use]
    pub fn gt<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x > value)
    }

    #[must_use]
    pub fn le<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x <= value)
    }

    #[must_use]
    pub fn ge<T>(self, value: T) -> Die<bool>
    where
        T: Into<K>,
    {
        let value = value.into();
        self.map(|x| x >= value)
    }
}

impl<K> Die<K>
where
    K: Clone + Ord,
{
    #[must_use]
    pub fn max(&self, rhs: &Self) -> Self {
        self.combine_with(rhs, |x, y| x.max(y).clone())
    }

    #[must_use]
    pub fn min(&self, rhs: &Self) -> Self {
        self.combine_with(rhs, |x, y| x.min(y).clone())
    }
}

impl<K> Die<K>
where
    K: Clone,
{
    pub fn sum<V, U>(&self, rhs: U) -> Die<<K as Add<V>>::Output>
    where
        V: Clone,
        K: Add<V, Output: Clone + Ord>,
        U: Borrow<Die<V>>,
    {
        self.combine_with(rhs, |x, y| x.clone() + y.clone())
    }
}

impl<K> Die<K> {
    #[must_use]
    pub fn shift<T>(self, value: T) -> Die<<T as Add<K>>::Output>
    where
        T: Copy + Add<K>,
    {
        self.biect_map(move |x| value + x)
    }

    #[must_use]
    pub fn kmul<T>(self, value: T) -> Die<<T as Mul<K>>::Output>
    where
        T: Copy + Mul<K>,
    {
        self.biect_map(move |x| value * x)
    }

    #[must_use]
    pub fn kdiv<T>(self, value: T) -> Die<<K as Div<T>>::Output>
    where
        T: Copy,
        K: Div<T>,
        <K as Div<T>>::Output: Ord,
    {
        self.map(move |x| x / value)
    }
}

impl<K> Die<K>
where
    K: TryInto<f64>,
    <K as TryInto<f64>>::Error: Clone + Debug,
{
    pub fn fdiv(self, value: f64) -> core::result::Result<Die<f64>, <K as TryInto<f64>>::Error> {
        self.try_biect_map(|x| x.try_into().map(|y| y / value))
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
    K: Copy + Ord + Sum,
    T: Into<usize>,
{
    type Output = Die<K>;

    fn mul(self, rhs: T) -> Self::Output {
        let dice: Vec<_> = std::iter::repeat_n(self, rhs.into()).collect();
        Die::combine(dice.as_slice(), |x| x.iter().map(|&&x| x).sum())
    }
}

impl<T, K> Mul<T> for &Die<K>
where
    K: Copy + Ord + Sum,
    T: Into<usize>,
{
    type Output = Die<K>;

    fn mul(self, rhs: T) -> Self::Output {
        let dice: Vec<_> = std::iter::repeat_n(self, rhs.into()).collect();
        Die::combine(dice.as_slice(), |x| x.iter().map(|&&x| x).sum())
    }
}
