use core::f64;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::{FusedIterator, IntoIterator, Zip};
use std::num::NonZeroU16;
use std::vec;

use itertools::Itertools;
use num::rational::Ratio;
use num::ToPrimitive;
use rand::{Rng, RngCore};

use crate::approx::Approx;
use crate::util::{
    die_map, BigInt, BigRatio, DieMap, Entry, Key, OverflowError, OverflowResult, Value,
    DIRECT_MAX_ITERATIONS,
};
use crate::{Die, Iter};

impl<K> Die<K>
where
    K: Clone + Copy + Ord,
{
    #[must_use]
    pub fn new<T, V>(values: Vec<(T, V)>) -> Self
    where
        T: Into<K>,
        V: Into<Value>,
    {
        let mut denom = 0;
        let mut keys = Vec::with_capacity(values.len());
        let mut outcomes = Vec::with_capacity(values.len());
        for (k, v) in values {
            let k = k.into();
            let v = v.into();
            if v == 0 {
                continue;
            }
            denom += v;
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
    pub fn uniform_values<T>(keys: Vec<T>) -> Self
    where
        T: Into<K>,
    {
        let c = 1;
        let n = keys.len();
        Self {
            denom: Value::try_from(n).unwrap(),
            keys: keys.into_iter().map(Into::into).collect(),
            outcomes: vec![c; n],
        }
    }

    #[must_use]
    pub fn single(key: K) -> Self {
        Self {
            denom: 1,
            keys: vec![key.into()],
            outcomes: vec![1],
        }
    }

    #[must_use]
    pub fn sample<G>(&self, rng: &mut G) -> K
    where
        G: RngCore,
    {
        let v = rng.gen_range(0..self.denom);
        let mut pos = 0;
        for (k, c) in self {
            pos += c;
            if v < pos {
                return k;
            }
        }
        unreachable!()
    }

    #[must_use]
    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            die: self,
            index: 0,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Value] {
        &self.outcomes
    }

    #[must_use]
    pub fn denom(&self) -> Value {
        self.denom
    }

    #[must_use]
    pub fn mode(&self) -> Vec<K> {
        self.iter()
            .max_set_by(|(_, x), (_, y)| x.cmp(y))
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }

    #[must_use]
    pub fn map<F, O>(self, op: F) -> Self
    where
        F: Fn(K) -> O,
        O: Into<K>,
    {
        let mut outcomes = die_map();
        for (k, c) in self.keys.into_iter().zip(self.outcomes) {
            match outcomes.entry(op(k).into()) {
                Entry::Vacant(e) => {
                    e.insert(c);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += c;
                }
            }
        }

        Die::from_map(self.denom, outcomes)
    }

    #[must_use]
    pub fn eval_n<F>(self, n: usize, op: F) -> Result<Self, K::Error>
    where
        F: Fn(K, K) -> K,
        K: TryInto<f64>,
    {
        let mut out = HashMap::new();
        out.insert(1usize, self);
        Self::eval_n_step(n, op, &mut out);
        Ok(out.remove(&n).unwrap())
    }

    fn eval_n_step<F>(n: usize, mut op: F, out: &mut HashMap<usize, Self>) -> Result<F, K::Error>
    where
        F: Fn(K, K) -> K,
        K: TryInto<f64>,
    {
        match n {
            1 => {}
            x if x % 2 == 1 => {
                let m = n - 1;
                if !out.contains_key(&m) {
                    op = Self::eval_n_step(m, op, out)?;
                }
                let dm = out.get(&m).unwrap();
                let d1 = out.get(&1).unwrap();
                let dn = Self::eval([dm, d1], |x| op(x[0], x[1]))?;
                out.insert(n, dn);
            }
            _ => {
                let m = n / 2;
                if !out.contains_key(&m) {
                    op = Self::eval_n_step(m, op, out)?;
                }
                let dm = out.get(&m).unwrap();
                let dn = Self::eval([dm, dm], |x| op(x[0], x[1]))?;
                out.insert(n, dn);
            }
        }
        Ok(op)
    }

    #[must_use]
    pub fn eval<L, D, O, F>(dice: L, op: F) -> Result<Die<O>, O::Error>
    where
        O: Clone + Copy + Ord + TryInto<f64>,
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[K]) -> O,
    {
        match (Self::iterations_of(&dice), Self::denom_of(&dice)) {
            (_, None) => Self::eval_approx(Approx::default(), dice, op),
            (Some(i), _) if i > DIRECT_MAX_ITERATIONS => {
                Self::eval_approx(Approx::default(), dice, op)
            }
            (_, Some(d)) => Ok(Self::eval_exact_impl(d, dice, op)),
        }
    }

    pub fn eval_exact<L, D, O, F>(dice: L, op: F) -> OverflowResult<Die<O>>
    where
        O: Clone + Copy + Ord,
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[K]) -> O,
    {
        Ok(Self::eval_exact_impl(
            Self::denom_of(&dice).ok_or(OverflowError)?,
            dice,
            op,
        ))
    }

    #[must_use]
    pub fn eval_approx<L, D, O, F, G>(
        mut approx: Approx<G>,
        dice: L,
        op: F,
    ) -> Result<Die<O>, O::Error>
    where
        O: Clone + Copy + Ord + TryInto<f64>,
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[K]) -> O,
        G: RngCore,
    {
        let dice = dice.borrow();
        approx.approximate(|rng| {
            let x: Vec<_> = dice.iter().map(|x| x.borrow().sample(rng)).collect();
            op(x.as_slice())
        })
    }

    #[must_use]
    fn eval_exact_impl<L, D, O, F>(denom: Value, dice: L, op: F) -> Die<O>
    where
        O: Clone + Copy + Ord,
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[K]) -> O,
    {
        let dice = dice.borrow();
        let mut outcomes = die_map();
        let mut key = Vec::with_capacity(dice.len());

        for p in dice
            .iter()
            .map(|x| x.borrow().iter())
            .multi_cartesian_product()
        {
            key.clear();
            let mut count = 1;
            for (k, c) in p {
                key.push(k);
                count *= c;
            }
            match outcomes.entry(op(key.as_slice())) {
                Entry::Vacant(e) => {
                    e.insert(count);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += count;
                }
            }
        }

        Die::<O>::from_map(denom, outcomes)
    }

    #[must_use]
    pub(crate) fn from_map(denom: Value, value: DieMap<K>) -> Self {
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

    fn denom_of<L, D>(dice: &L) -> Option<Value>
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
    {
        dice.borrow()
            .iter()
            .try_fold(1 as Value, |acc, x| acc.checked_mul(x.borrow().denom))
    }

    fn iterations_of<L, D>(dice: &L) -> Option<usize>
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
    {
        dice.borrow()
            .iter()
            .try_fold(1usize, |acc, x| acc.checked_mul(x.borrow().keys.len()))
    }
}

impl<K> Die<K>
where
    K: Clone + Copy + Ord + Into<BigInt>,
{
    #[must_use]
    pub fn probabilities(&self) -> Vec<(K, BigRatio)> {
        let d = BigInt::from(self.denom);
        self.iter()
            .map(|(k, c)| (k, Ratio::new(c.into(), d.clone())))
            .collect()
    }

    #[must_use]
    pub fn probabilities_f64(&self) -> Vec<(K, f64)> {
        let d = BigInt::from(self.denom);
        self.iter()
            .map(|(k, c)| {
                (
                    k,
                    Ratio::new(c.into(), d.clone()).to_f64().unwrap_or(f64::NAN),
                )
            })
            .collect()
    }

    #[must_use]
    pub fn mean(&self) -> BigRatio {
        let c = self
            .iter()
            .map(|(k, c)| Self::map_mean(k, c))
            .reduce(|acc, x| acc + x)
            .unwrap();
        BigRatio::new(c, self.denom.into())
    }

    #[must_use]
    pub fn mean_f64(&self) -> f64 {
        self.mean().to_f64().unwrap_or(f64::NAN)
    }

    #[must_use]
    pub fn variance(&self) -> BigRatio {
        let m = self.mean();
        let c = self
            .iter()
            .map(|(k, c)| Self::map_variance(&m, k, c))
            .reduce(|acc, x| acc + x)
            .unwrap();
        c / BigInt::from(self.denom)
    }

    #[must_use]
    pub fn variance_f64(&self) -> f64 {
        self.variance().to_f64().unwrap_or(f64::NAN)
    }

    #[must_use]
    pub fn stddev_f64(&self) -> f64 {
        self.variance_f64().sqrt()
    }

    fn map_mean(k: K, c: Value) -> BigInt {
        let k = k.into();
        let c = BigInt::from(c);
        k * c
    }

    fn map_variance(mean: &BigRatio, k: K, c: Value) -> BigRatio {
        let k = Ratio::new(k.into(), 1.into());
        let c = BigInt::from(c);
        (k - mean).pow(2) * c
    }
}

impl Die {
    #[must_use]
    pub fn uniform(size: NonZeroU16) -> Die {
        let size = size.get();
        Die {
            denom: Value::from(size),
            keys: (1..=Key::from(size)).collect(),
            outcomes: vec![1; size as usize],
        }
    }
}

impl IntoIterator for Die {
    type Item = (Key, Value);

    type IntoIter = Zip<vec::IntoIter<Key>, vec::IntoIter<Value>>;

    fn into_iter(self) -> Self::IntoIter {
        let keys = self.keys;
        let outcomes = self.outcomes;
        keys.into_iter().zip(outcomes)
    }
}

impl<'a, K> IntoIterator for &'a Die<K>
where
    K: Clone + Copy + Ord,
{
    type Item = (K, Value);

    type IntoIter = Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K> Iterator for Iter<'a, K>
where
    K: Clone + Copy + Ord,
{
    type Item = (K, Value);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.die.keys.len() {
            None
        } else {
            let item = (self.die.keys[self.index], self.die.outcomes[self.index]);
            self.index += 1;
            Some(item)
        }
    }
}

impl<'a, K> FusedIterator for Iter<'a, K> where K: Clone + Copy + Ord {}
