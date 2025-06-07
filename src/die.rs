use core::f64;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::{IntoIterator, Zip};
use std::num::NonZeroU16;
use std::{slice, vec};

use itertools::Itertools;
use num::rational::Ratio;
use num::{Integer, ToPrimitive};
use rand::{thread_rng, Rng, RngCore};

use crate::approx::Approx;
use crate::util::{
    die_map, BigInt, BigRatio, DieMap, Entry, Key, OverflowError, OverflowResult, Value,
    DIRECT_MAX_ITERATIONS,
};
use crate::{Die, EvalStrategy};

pub type Iter<'a> = Zip<slice::Iter<'a, Key>, slice::Iter<'a, Value>>;
pub type IntoIter = Zip<vec::IntoIter<Key>, vec::IntoIter<Value>>;

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
    pub fn sample_rng<G>(&self, rng: &mut G) -> Key
    where
        G: RngCore,
    {
        let v = rng.gen_range(0..self.denom);
        let mut pos = 0;
        for (&k, c) in self {
            pos += c;
            if v < pos {
                return k;
            }
        }
        unreachable!()
    }

    #[must_use]
    pub fn sample_n_rng<G>(&self, n: usize, rng: &mut G) -> Vec<Key>
    where
        G: RngCore,
    {
        (0..n).map(|_| self.sample_rng(rng)).collect()
    }

    #[must_use]
    pub fn sample(&self) -> Key {
        let mut rng = thread_rng();
        self.sample_rng(&mut rng)
    }

    #[must_use]
    pub fn sample_n(&self, n: usize) -> Vec<Key> {
        let mut rng = thread_rng();
        self.sample_n_rng(n, &mut rng)
    }

    pub fn iter(&self) -> Iter<'_> {
        self.keys.iter().zip(self.outcomes.iter())
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
    pub fn mode(&self) -> Vec<Key> {
        self.iter()
            .max_set_by(|(_, x), (_, y)| x.cmp(y))
            .into_iter()
            .map(|(&k, _)| k)
            .collect()
    }

    #[must_use]
    pub fn probabilities(&self) -> Vec<(Key, BigRatio)> {
        let d = BigInt::from(self.denom);
        self.iter()
            .map(|(&k, &c)| (k, Ratio::new(c.into(), d.clone())))
            .collect()
    }

    #[must_use]
    pub fn probabilities_f64(&self) -> Vec<(Key, f64)> {
        let d = BigInt::from(self.denom);
        self.iter()
            .map(|(&k, &c)| {
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
            .map(|(&k, &c)| Self::map_mean(k, c))
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
            .map(|(&k, &c)| Self::map_variance(&m, k, c))
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
    pub fn eval_n<F>(self, n: usize, op: F) -> Die
    where
        F: Fn(Key, Key) -> Key,
    {
        let mut out = HashMap::new();
        out.insert(1usize, self);
        Self::eval_n_step(n, op, &mut out);
        out.remove(&n).unwrap()
    }

    fn eval_n_step<F>(n: usize, mut op: F, out: &mut HashMap<usize, Self>) -> F
    where
        F: Fn(Key, Key) -> Key,
    {
        match n {
            1 => {}
            x if x % 2 == 1 => {
                let m = n - 1;
                if !out.contains_key(&m) {
                    op = Self::eval_n_step(m, op, out);
                }
                let dm = out.get(&m).unwrap();
                let d1 = out.get(&1).unwrap();
                let dn = Self::eval([dm, d1], |x| op(x[0], x[1]));
                out.insert(n, dn);
            }
            _ => {
                let m = n / 2;
                if !out.contains_key(&m) {
                    op = Self::eval_n_step(m, op, out);
                }
                let dm = out.get(&m).unwrap();
                let dn = Self::eval([dm, dm], |x| op(x[0], x[1]));
                out.insert(n, dn);
            }
        }
        op
    }

    pub fn eval_with_strategy<L, D, F>(
        strategy: EvalStrategy,
        dice: L,
        op: F,
    ) -> OverflowResult<Die>
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
    {
        match strategy {
            EvalStrategy::Any => Ok(Self::eval(dice, op)),
            EvalStrategy::Approximate => Ok(Self::eval_approx(dice, op)),
            EvalStrategy::Exact => Self::eval_exact(dice, op),
        }
    }

    #[must_use]
    pub fn eval<L, D, F>(dice: L, op: F) -> Die
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
    {
        match (Self::iterations_of(&dice), Self::denom_of(&dice)) {
            (_, None) => Self::eval_approx(dice, op),
            (Some(i), _) if i > DIRECT_MAX_ITERATIONS => Self::eval_approx(dice, op),
            (_, Some(d)) => Self::eval_exact_impl(d, dice, op),
        }
    }

    pub fn eval_exact<L, D, F>(dice: L, op: F) -> OverflowResult<Die>
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
    {
        Ok(Self::eval_exact_impl(
            Self::denom_of(&dice).ok_or(OverflowError)?,
            dice,
            op,
        ))
    }

    #[must_use]
    pub fn eval_approx_custom<L, D, F, G>(mut approx: Approx<G>, dice: L, op: F) -> Die
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
        G: RngCore,
    {
        let dice = dice.borrow();
        approx.approximate(|rng| {
            let x: Vec<_> = dice.iter().map(|x| x.borrow().sample_rng(rng)).collect();
            op(x.as_slice())
        })
    }

    #[must_use]
    pub fn eval_approx<L, D, F>(dice: L, op: F) -> Die
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
    {
        Self::eval_approx_custom(Approx::default(), dice, op)
    }

    #[must_use]
    fn eval_exact_impl<L, D, F>(denom: Value, dice: L, op: F) -> Die
    where
        L: Borrow<[D]>,
        D: Borrow<Self>,
        F: Fn(&[Key]) -> Key,
    {
        let dice = dice.borrow();
        let mut outcomes = die_map();
        let mut key = Vec::with_capacity(dice.len());

        for p in dice
            .iter()
            .map(|x| x.borrow().iter().map(|(&k, &c)| (k, c)))
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
    pub(crate) fn from_map(mut denom: Value, value: DieMap) -> Self {
        let mut keys = Vec::with_capacity(value.len());
        let mut outcomes = Vec::with_capacity(value.len());
        let mut gcd = denom;
        for (k, v) in value {
            gcd = gcd.gcd(&v);
            keys.push(k);
            outcomes.push(v);
        }
        if gcd != 1 {
            outcomes.iter_mut().for_each(|x| *x /= gcd);
            denom /= gcd;
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

impl IntoIterator for Die {
    type Item = (Key, Value);
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.keys.into_iter().zip(self.outcomes)
    }
}

impl<'a> IntoIterator for &'a Die {
    type Item = (&'a Key, &'a Value);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
