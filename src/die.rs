use core::f64;
use std::borrow::Borrow;
use std::iter::{FusedIterator, IntoIterator, Zip};
use std::num::NonZeroU16;
use std::vec;

use itertools::Itertools;
use num::rational::Ratio;
use num::ToPrimitive;
use rand::{Rng, RngCore};

use crate::approx::Approx;
use crate::util::{
    die_map, BigInt, BigRatio, DieList, DieMap, Entry, Key, OverflowError, OverflowResult, Value,
    DIRECT_MAX_DENOM,
};
use crate::{Die, Iter};

impl Die {
    #[must_use]
    pub fn new<K, V>(values: Vec<(K, V)>) -> Self
    where
        K: Into<Key>,
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
    pub fn uniform(size: NonZeroU16) -> Self {
        let size = size.get();
        Self {
            denom: Value::from(size),
            keys: (1..=Key::from(size)).collect(),
            outcomes: vec![1; size as usize],
        }
    }

    #[must_use]
    pub fn uniform_values<K>(keys: Vec<K>) -> Self
    where
        K: Into<Key>,
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
    pub fn single<K>(key: K) -> Self
    where
        K: Into<Key>,
    {
        Self {
            denom: 1,
            keys: vec![key.into()],
            outcomes: vec![1],
        }
    }

    #[must_use]
    pub fn sample<G>(&self, rng: &mut G) -> Key
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
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            die: self,
            index: 0,
        }
    }

    #[must_use]
    pub fn keys(&self) -> &[Key] {
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
    pub fn map<F, O>(self, op: F) -> Die
    where
        F: Fn(Key) -> O,
        O: Into<Key>,
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
    pub fn mode(&self) -> Vec<Key> {
        self.iter()
            .max_set_by(|(_, x), (_, y)| x.cmp(y))
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }

    #[must_use]
    pub fn probabilities(&self) -> Vec<(Key, BigRatio)> {
        let d = BigInt::from(self.denom);
        self.iter()
            .map(|(k, c)| (k, Ratio::new(c.into(), d.clone())))
            .collect()
    }

    #[must_use]
    pub fn probabilities_f64(&self) -> Vec<(Key, f64)> {
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

    fn map_mean(k: Key, c: Value) -> BigInt {
        let k = BigInt::from(k);
        let c = BigInt::from(c);
        k * c
    }

    fn map_variance(mean: &BigRatio, k: Key, c: Value) -> BigRatio {
        let k = Ratio::new(BigInt::from(k), 1.into());
        let c = BigInt::from(c);
        (k - mean).pow(2) * c
    }

    #[must_use]
    pub(crate) fn eval<F>(dice: DieList, op: F) -> Die
    where
        F: Fn(&[Key]) -> Key,
    {
        let denom = dice
            .iter()
            .try_fold(1 as Value, |acc, x| acc.checked_mul(x.denom));
        match denom {
            None => Self::eval_approx(Approx::default(), &dice, op),
            Some(x) if x > Value::from(DIRECT_MAX_DENOM) => {
                Self::eval_approx(Approx::default(), &dice, op)
            }
            Some(x) => Self::eval_exact_impl(x, dice, op),
        }
    }

    pub(crate) fn eval_exact<F>(dice: DieList, op: F) -> OverflowResult<Die>
    where
        F: Fn(&[Key]) -> Key,
    {
        let Some(denom) = dice
            .iter()
            .try_fold(1 as Value, |acc, x| acc.checked_mul(x.borrow().denom))
        else {
            return Err(OverflowError);
        };
        Ok(Self::eval_exact_impl(denom, dice, op))
    }

    #[must_use]
    pub(crate) fn eval_approx<F, G>(mut approx: Approx<G>, dice: &DieList, op: F) -> Die
    where
        F: Fn(&[Key]) -> Key,
        G: RngCore,
    {
        approx.approximate(|rng| {
            let x: Vec<_> = dice.iter().map(|x| x.sample(rng)).collect();
            op(x.as_slice())
        })
    }

    #[must_use]
    fn eval_exact_impl<F>(denom: Value, dice: DieList, op: F) -> Die
    where
        F: Fn(&[Key]) -> Key,
    {
        let mut outcomes = die_map();
        let mut key = Vec::with_capacity(dice.len());
        for p in dice
            .into_iter()
            .map(IntoIterator::into_iter)
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
        Die::from_map(denom, outcomes)
    }

    #[must_use]
    pub(crate) fn from_map(denom: Value, value: DieMap) -> Self {
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

impl<'a> IntoIterator for &'a Die {
    type Item = (Key, Value);

    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Iterator for Iter<'_> {
    type Item = (Key, Value);

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

impl FusedIterator for Iter<'_> {}
