use std::borrow::Borrow;
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::num::NonZeroU16;
#[cfg(feature = "parallel")]
use std::sync::{mpsc, Arc};

use itertools::Itertools;
use num::rational::Ratio;
use num::ToPrimitive;
use rand::{Rng, RngCore};
use thiserror::Error;

use crate::approx::Approx;
use crate::util::{
    die_map, BigInt, BigRatio, DieMap, Entry, Key, Probability, Value, DIRECT_MAX_DENOM,
};

#[derive(Debug, Clone)]
pub struct Die {
    denom: Value,
    keys: Vec<Key>,
    outcomes: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct DieIntoIter {
    die: Die,
    index: usize,
}

#[derive(Debug, Clone)]
pub struct DieIter<'a> {
    die: &'a Die,
    index: usize,
}

#[derive(Debug, Clone, Error)]
#[error("overflow in probabilities")]
pub struct OverflowError;

pub type OverflowResult<T> = Result<T, OverflowError>;

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
    pub fn iter(&self) -> DieIter<'_> {
        DieIter {
            die: self,
            index: 0,
        }
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
    pub fn probabilities<T>(self) -> Option<Vec<(T, f64)>>
    where
        T: From<Key>,
    {
        let d = BigInt::from(self.denom);
        self.keys
            .into_iter()
            .zip(self.outcomes)
            .map(|(k, c)| {
                Ratio::new(c.into(), d.clone())
                    .to_f64()
                    .map(|v| (k.into(), v))
            })
            .collect()
    }

    #[must_use]
    pub fn mean(&self) -> Option<Probability> {
        self.mean_ratio().to_f64()
    }

    fn mean_ratio(&self) -> BigRatio {
        let c = self
            .iter()
            .map(|(k, c)| Self::map_mean(k, c))
            .reduce(|acc, x| acc + x)
            .unwrap();
        BigRatio::new(c, self.denom.into())
    }

    #[must_use]
    pub fn variance(&self) -> Option<Probability> {
        self.variance_ratio().to_f64()
    }

    fn variance_ratio(&self) -> BigRatio {
        let m = self.mean_ratio();
        let c = self
            .iter()
            .map(|(k, c)| Self::map_variance(&m, k, c))
            .reduce(|acc, x| acc + x)
            .unwrap();
        c / BigInt::from(self.denom)
    }

    #[must_use]
    pub fn stddev(&self) -> Option<Probability> {
        self.variance().map(Probability::sqrt)
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
    pub fn combine<F, O, Q, V>(dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O + Send + Sync,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        let denom = dice
            .borrow()
            .iter()
            .try_fold(1 as Value, |acc, x| acc.checked_mul(x.borrow().denom));
        match denom {
            None => Self::combine_approx(Approx::default(), dice, op),
            Some(x) if x > Value::from(DIRECT_MAX_DENOM) => {
                Self::combine_approx(Approx::default(), dice, op)
            }
            Some(x) => Self::combine_direct(x, dice, op),
        }
    }

    pub fn try_combine<F, O, Q, V>(dice: V, op: F) -> OverflowResult<Die>
    where
        F: Fn(&[Key]) -> O + Send + Sync,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        let Some(denom) = dice
            .borrow()
            .iter()
            .try_fold(1 as Value, |acc, x| acc.checked_mul(x.borrow().denom)) else {
                return Err(OverflowError);
            };
        Ok(Self::combine_direct(denom, dice, op))
    }

    #[must_use]
    pub(crate) fn combine_approx<F, O, Q, V, G>(mut approx: Approx<G>, dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
        G: RngCore,
    {
        let dice = dice.borrow();
        approx.approximate(|rng| {
            let x: Vec<_> = dice.iter().map(|x| x.borrow().sample(rng)).collect();
            op(x.as_slice()).into()
        })
    }

    #[cfg(not(feature = "parallel"))]
    #[inline]
    #[must_use]
    pub(crate) fn combine_direct<F, O, Q, V>(denom: Value, dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        Self::combine_direct_single_thread(denom, dice, op)
    }

    #[cfg(feature = "parallel")]
    #[inline]
    #[must_use]
    pub(crate) fn combine_direct<F, O, Q, V>(denom: Value, dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O + Send + Sync,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        use crate::util::PARALLEL_MIN_DENOM;

        if denom < PARALLEL_MIN_DENOM {
            Self::combine_direct_single_thread(denom, dice, op)
        } else {
            Self::combine_direct_parallel(denom, dice, op)
        }
    }

    #[must_use]
    pub(crate) fn combine_direct_single_thread<F, O, Q, V>(denom: Value, dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
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
            match outcomes.entry(op(key.as_slice()).into()) {
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

    #[cfg(feature = "parallel")]
    #[must_use]
    pub(crate) fn combine_direct_parallel<F, O, Q, V>(denom: Value, dice: V, op: F) -> Die
    where
        F: Fn(&[Key]) -> O + Send + Sync,
        O: Into<Key>,
        Q: Borrow<Self>,
        V: Borrow<[Q]>,
    {
        use crate::util::PARALLEL_CHUNK_SIZE;

        let dice = dice.borrow();

        let jobs = rayon::current_num_threads();
        let n = dice.len();
        let m = dice
            .iter()
            .map(|x| x.borrow().keys.len())
            .product::<usize>();
        let chunk_size = PARALLEL_CHUNK_SIZE.max(m / jobs);

        let op = Arc::new(op);
        let combinations = dice
            .iter()
            .map(|x| x.borrow().to_owned().into_iter())
            .multi_cartesian_product();
        let outcomes = rayon::scope(move |s| {
            let (tx, rx) = mpsc::channel();
            for chunk in &combinations.chunks(chunk_size) {
                let op = op.clone();
                let tx = tx.clone();
                let chunk = chunk.collect();
                s.spawn(move |_| {
                    tx.send(Self::combine_direct_parallel_chunk(n, &op, chunk))
                        .ok();
                });
            }
            drop(tx);

            let mut outcomes = die_map();
            while let Ok(part) = rx.recv() {
                for (k, v) in part {
                    match outcomes.entry(k) {
                        Entry::Vacant(e) => {
                            e.insert(v);
                        }
                        Entry::Occupied(mut e) => {
                            *e.get_mut() += v;
                        }
                    }
                }
            }

            outcomes
        });

        Die::from_map(denom, outcomes)
    }

    #[cfg(feature = "parallel")]
    #[must_use]
    fn combine_direct_parallel_chunk<F, O>(
        n: usize,
        op: &Arc<F>,
        chunk: Vec<Vec<(Key, Value)>>,
    ) -> DieMap
    where
        F: Fn(&[Key]) -> O,
        O: Into<Key>,
    {
        let mut outcomes = die_map();
        let mut key = vec![0; n];
        for p in chunk {
            let mut count = 1;
            for (i, (k, c)) in p.into_iter().enumerate() {
                key[i] = k;
                count *= c;
            }
            match outcomes.entry(op(key.as_slice()).into()) {
                Entry::Vacant(e) => {
                    e.insert(count);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += count;
                }
            }
        }
        outcomes
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

    type IntoIter = DieIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        DieIntoIter {
            die: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Die {
    type Item = (Key, Value);

    type IntoIter = DieIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Iterator for DieIntoIter {
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

impl FusedIterator for DieIntoIter {}

impl Iterator for DieIter<'_> {
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

impl FusedIterator for DieIter<'_> {}
