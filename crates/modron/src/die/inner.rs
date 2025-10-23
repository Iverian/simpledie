use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::vec;

use itertools::Itertools;
use rand::{Rng, RngCore};

use super::{DieLike, Iter};
use crate::value::{ComputableValue, ComputedValue, DefaultValue, Value};
use crate::{Approx, Error, Map, Outcome, Result, DIRECT_MAX_ITERATIONS};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DieInner<T = DefaultValue>
where
    T: Value,
{
    denom: Outcome,
    outcomes: Vec<Outcome>,
    values: Vec<T>,
}

enum Strategy {
    Direct(Outcome),
    Approx,
}

impl<T> DieInner<T>
where
    T: Value,
{
    #[must_use]
    pub fn scalar(value: T) -> Self {
        Self {
            values: vec![value],
            outcomes: vec![1],
            denom: 1,
        }
    }

    #[must_use]
    pub fn uniform<I>(values: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let values: Vec<_> = values.into_iter().sorted().collect();
        let n = values.len();
        Self {
            values,
            outcomes: vec![1; n],
            denom: n as Outcome,
        }
    }

    #[must_use]
    pub fn denom(&self) -> Outcome {
        self.denom
    }

    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.values.iter().zip(self.outcomes.iter())
    }

    #[must_use]
    pub fn modes(&self) -> Vec<&T> {
        self.values
            .iter()
            .zip(&self.outcomes)
            .max_set_by_key(|(_, o)| **o)
            .into_iter()
            .map(|(v, _)| v)
            .collect()
    }

    #[must_use]
    pub fn mode(&self) -> Option<&T> {
        self.values
            .iter()
            .zip(&self.outcomes)
            .max_by_key(|(_, o)| **o)
            .map(|(v, _)| v)
    }

    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.outcomes
            .iter()
            .map(|x| (*x as f64) / self.denom as f64)
            .collect_vec()
    }

    #[must_use]
    pub fn new(map: Map<T>, mut denom: Outcome) -> Self {
        let mut values = Vec::with_capacity(map.len());
        let mut outcomes = Vec::with_capacity(map.len());
        let mut acc = denom;
        for (value, outcome) in map.into_iter().sorted_by(|(v1, _), (v2, _)| v1.cmp(v2)) {
            acc = gcd(acc, outcome);
            values.push(value);
            outcomes.push(outcome);
        }
        if acc != 1 {
            outcomes.iter_mut().for_each(|x| *x /= acc);
            denom /= acc;
        }
        Self {
            denom,
            values,
            outcomes,
        }
    }

    #[must_use]
    pub fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        let x = rng.random_range(0..self.denom);
        let mut pos = 0;
        for (v, o) in self.values.iter().zip(&self.outcomes) {
            pos += o;
            if x < pos {
                return v;
            }
        }
        unreachable!()
    }

    #[must_use]
    pub fn map<O, F>(&self, f: F) -> DieInner<O>
    where
        O: Value,
        F: Fn(&T) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            let o = map.entry(f(v1)).or_default();
            *o += o1;
        }

        DieInner::new(map, self.denom)
    }

    #[must_use]
    pub fn apply_two<T2, O, F>(&self, d2: &DieInner<T2>, f: F) -> DieInner<O>
    where
        T2: Value,
        O: Value,
        F: Fn(&T, &T2) -> O,
    {
        let iter = self.values.len().checked_mul(d2.values.len());
        let denom = self.denom.checked_mul(d2.denom);
        match Strategy::select(iter, denom) {
            Strategy::Direct(denom) => self.apply_two_direct(denom, d2, f),
            Strategy::Approx => self.apply_two_approx(d2, f),
        }
    }

    #[must_use]
    fn apply_two_approx<T2, O, F>(&self, d2: &DieInner<T2>, f: F) -> DieInner<O>
    where
        T2: Value,
        O: Value,
        F: Fn(&T, &T2) -> O,
    {
        Approx::default().eval_inner(|rng| f(self.sample_rng(rng), d2.sample_rng(rng)))
    }

    #[must_use]
    fn apply_two_direct<T2, O, F>(&self, denom: Outcome, d2: &DieInner<T2>, f: F) -> DieInner<O>
    where
        T2: Value,
        O: Value,
        F: Fn(&T, &T2) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            for (v2, o2) in d2.iter() {
                let o = map.entry(f(v1, v2)).or_default();
                *o += o1 * o2;
            }
        }

        DieInner::new(map, denom)
    }

    #[must_use]
    pub fn apply_three<T2, T3, O, F>(
        &self,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        O: Value,
        F: Fn(&T, &T2, &T3) -> O,
    {
        let iter = self
            .values
            .len()
            .checked_mul(d2.values.len())
            .and_then(|x| x.checked_mul(d3.values.len()));
        let denom = self
            .denom
            .checked_mul(d2.denom)
            .and_then(|x| x.checked_mul(d3.denom));
        match Strategy::select(iter, denom) {
            Strategy::Direct(denom) => self.apply_three_direct(denom, d2, d3, f),
            Strategy::Approx => self.apply_three_approx(d2, d3, f),
        }
    }

    #[must_use]
    fn apply_three_approx<T2, T3, O, F>(
        &self,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        O: Value,
        F: Fn(&T, &T2, &T3) -> O,
    {
        Approx::default()
            .eval_inner(|rng| f(self.sample_rng(rng), d2.sample_rng(rng), d3.sample_rng(rng)))
    }

    #[must_use]
    fn apply_three_direct<T2, T3, O, F>(
        &self,
        denom: Outcome,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        O: Value,
        F: Fn(&T, &T2, &T3) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            for (v2, o2) in d2.iter() {
                for (v3, o3) in d3.iter() {
                    let o = map.entry(f(v1, v2, v3)).or_default();
                    *o += o1 * o2 * o3;
                }
            }
        }

        DieInner::new(map, denom)
    }

    #[must_use]
    pub fn apply_four<T2, T3, T4, O, F>(
        &self,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        d4: &DieInner<T4>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        T4: Value,
        O: Value,
        F: Fn(&T, &T2, &T3, &T4) -> O,
    {
        let iter = self
            .values
            .len()
            .checked_mul(d2.values.len())
            .and_then(|x| x.checked_mul(d3.values.len()))
            .and_then(|x| x.checked_mul(d4.values.len()));
        let denom = self
            .denom
            .checked_mul(d2.denom)
            .and_then(|x| x.checked_mul(d3.denom))
            .and_then(|x| x.checked_mul(d4.denom));
        match Strategy::select(iter, denom) {
            Strategy::Direct(denom) => self.apply_four_direct(denom, d2, d3, d4, f),
            Strategy::Approx => self.apply_four_approx(d2, d3, d4, f),
        }
    }

    #[must_use]
    fn apply_four_approx<T2, T3, T4, O, F>(
        &self,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        d4: &DieInner<T4>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        T4: Value,
        O: Value,
        F: Fn(&T, &T2, &T3, &T4) -> O,
    {
        Approx::default().eval_inner(|rng| {
            f(
                self.sample_rng(rng),
                d2.sample_rng(rng),
                d3.sample_rng(rng),
                d4.sample_rng(rng),
            )
        })
    }

    #[must_use]
    fn apply_four_direct<T2, T3, T4, O, F>(
        &self,
        denom: Outcome,
        d2: &DieInner<T2>,
        d3: &DieInner<T3>,
        d4: &DieInner<T4>,
        f: F,
    ) -> DieInner<O>
    where
        T2: Value,
        T3: Value,
        T4: Value,
        O: Value,
        F: Fn(&T, &T2, &T3, &T4) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            for (v2, o2) in d2.iter() {
                for (v3, o3) in d3.iter() {
                    for (v4, o4) in d4.iter() {
                        let o = map.entry(f(v1, v2, v3, v4)).or_default();
                        *o += o1 * o2 * o3 * o4;
                    }
                }
            }
        }

        DieInner::new(map, denom)
    }

    #[must_use]
    pub fn apply<I, Q, D, O, F>(dice: I, f: F) -> DieInner<O>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
        O: Value,
        F: Fn(&[&T]) -> O,
    {
        let iter = Self::iterations_of(&dice);
        let denom = Self::denom_of(&dice);
        match Strategy::select(iter, denom) {
            Strategy::Direct(denom) => Self::apply_direct(denom, dice, f),
            Strategy::Approx => Self::apply_approx(dice, f),
        }
    }

    #[must_use]
    fn apply_approx<I, Q, D, O, F>(dice: I, f: F) -> DieInner<O>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
        O: Value,
        F: Fn(&[&T]) -> O,
    {
        let dice = dice.borrow();
        let mut value = Vec::with_capacity(dice.len());
        Approx::default().eval_inner(|rng| {
            value.clear();
            for d in dice {
                value.push(d.borrow().sample_rng(rng));
            }
            f(&value)
        })
    }

    #[must_use]
    fn apply_direct<I, Q, D, O, F>(denom: Outcome, dice: I, f: F) -> DieInner<O>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
        O: Value,
        F: Fn(&[&T]) -> O,
    {
        let dice = dice.borrow();
        let mut map = Map::new();
        let mut value = Vec::with_capacity(dice.len());

        for p in dice
            .iter()
            .map(|x| {
                let x = x.borrow();
                x.values().iter().zip(x.outcomes())
            })
            .multi_cartesian_product()
        {
            value.clear();
            let mut outcome: Outcome = 1;
            for (v, o) in p {
                value.push(v);
                outcome *= o;
            }
            let o = map.entry(f(value.as_slice())).or_default();
            *o += outcome;
        }

        DieInner::new(map, denom)
    }

    pub fn combine<I, Q, D>(dice: I) -> Result<DieInner<Vec<T>>>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
    {
        let dice = dice.borrow();
        let total_len = Self::iterations_of(&dice).ok_or(Error)?;
        let denom = Self::denom_of(&dice).ok_or(Error)?;
        let value_len = dice.len();

        let mut values = Vec::with_capacity(total_len);
        let mut outcomes = Vec::with_capacity(total_len);

        for p in dice
            .iter()
            .map(|x| {
                let x = x.borrow();
                x.values().iter().zip(x.outcomes())
            })
            .multi_cartesian_product()
        {
            let mut value = Vec::with_capacity(value_len);
            let mut outcome: Outcome = 1;
            for (v, o) in p {
                value.push(v.clone());
                outcome *= o;
            }
            values.push(value);
            outcomes.push(outcome);
        }

        Ok(DieInner {
            denom,
            outcomes,
            values,
        })
    }

    #[must_use]
    pub fn fold<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        let mut die = self.apply_two(self, &f);
        for _ in 0..n.saturating_sub(2) {
            die = die.apply_two(self, &f);
        }

        die
    }

    #[must_use]
    pub fn fold_assoc<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        let mut cache = HashMap::new();
        cache.insert(2, self.apply_two(self, &f));

        let mut stack = vec![n];
        while let Some(&x) = stack.last() {
            if x % 2 == 0 {
                let m = x / 2;
                if let Some(d) = cache.get(&m) {
                    cache.insert(x, d.apply_two(d, &f));
                    stack.pop();
                } else {
                    stack.push(m);
                }
            } else {
                let m = x - 1;
                if let Some(d) = cache.get(&m) {
                    cache.insert(x, d.apply_two(self, &f));
                    stack.pop();
                } else {
                    stack.push(m);
                }
            }
        }

        cache.remove(&n).expect("nth repeat to be calculated")
    }

    #[must_use]
    pub fn explode<P, F>(&self, min: usize, max: usize, predicate: P, fold: F) -> Self
    where
        P: Fn(&[&T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        let iter = self.values.len().checked_pow(max as u32);
        let denom = self.denom.checked_pow(max as u32);
        match Strategy::select(iter, denom) {
            Strategy::Direct(denom) => self.explode_direct(denom, min, max, predicate, fold),
            Strategy::Approx => self.explode_approx(min, max, predicate, fold),
        }
    }

    #[must_use]
    fn explode_approx<P, F>(&self, min: usize, max: usize, p: P, f: F) -> Self
    where
        P: Fn(&[&T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        let mut value = Vec::with_capacity(max);
        Approx::default().eval_inner(|rng| {
            value.clear();
            value.push(self.sample_rng(rng));
            let mut result = value[0].clone();
            for _ in 1..min {
                let x = self.sample_rng(rng);
                result = f(&result, x);
                value.push(x);
            }
            for _ in min..max {
                if !p(&value) {
                    break;
                }
                let x = self.sample_rng(rng);
                result = f(&result, x);
                value.push(x);
            }
            result
        })
    }

    #[must_use]
    fn explode_direct<P, F>(
        &self,
        denom: Outcome,
        min: usize,
        max: usize,
        predicate: P,
        fold: F,
    ) -> Self
    where
        P: Fn(&[&T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        let m = self.values.len();
        let mut map = Map::new();

        let dice = Self::combine::<_, _, Self>(vec![self; min]).unwrap();
        let mut items = dice
            .values
            .iter()
            .zip(dice.outcomes)
            .map(|(v, o)| (fold_with(v, &fold), v.iter().collect_vec(), o))
            .collect_vec();

        for i in (min + 1)..=max {
            let trees = self.denom.pow((max - i) as u32);
            let mut new_items = Vec::with_capacity(items.len() * m);

            for (r1, v1, o1) in items {
                let mut negative_outcome = 0;

                for (v2, o2) in self.iter() {
                    let outcome = o1 * *o2;
                    let mut value = Vec::with_capacity(i);
                    value.extend(v1.iter().copied());
                    value.push(v2);
                    if predicate(&value) {
                        new_items.push((fold(&r1, v2), value, outcome));
                    } else {
                        negative_outcome += outcome;
                    }
                }

                if negative_outcome != 0 {
                    let o = map.entry(r1).or_default();
                    negative_outcome *= trees;
                    *o += negative_outcome;
                }
            }

            items = new_items;
        }

        for (v1, _, o1) in items {
            let o = map.entry(v1).or_default();
            *o += o1;
        }

        Self::new(map, denom)
    }

    #[must_use]
    fn denom_of<I, Q, D>(dice: &I) -> Option<Outcome>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
    {
        dice.borrow()
            .iter()
            .try_fold(1 as Outcome, |acc, x| acc.checked_mul(x.borrow().denom()))
    }

    #[must_use]
    fn iterations_of<I, Q, D>(dice: &I) -> Option<usize>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
    {
        dice.borrow()
            .iter()
            .try_fold(1usize, |acc, x| acc.checked_mul(x.borrow().values().len()))
    }
}

impl<T> DieInner<T>
where
    T: ComputableValue,
{
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean_computed(self.values.iter().map(|x| x.compute_f64()))
    }

    #[must_use]
    pub fn variance(&self) -> f64 {
        let computed = self.computed_f64();
        let mean = self.mean_computed(computed.iter());

        computed
            .iter()
            .zip(&self.outcomes)
            .fold(0.0, |mut acc, (value, outcome)| {
                acc += (*value - mean).powi(2) * (*outcome as f64) / (self.denom as f64);
                acc
            })
    }

    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[must_use]
    pub fn computed_values(&self) -> Vec<ComputedValue> {
        self.values.iter().map(|x| x.compute()).collect()
    }

    fn computed_f64(&self) -> Vec<f64> {
        self.values.iter().map(|x| x.compute_f64()).collect()
    }

    fn mean_computed<I, F>(&self, computed: I) -> f64
    where
        I: Iterator<Item = F>,
        F: Borrow<f64>,
    {
        computed
            .zip(&self.outcomes)
            .fold(0.0, |mut acc, (value, outcome)| {
                acc += value.borrow() * (*outcome as f64) / (self.denom as f64);
                acc
            })
    }
}

impl<T> DieLike<T> for DieInner<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome {
        self.denom
    }
    fn values(&self) -> &[T] {
        &self.values
    }

    fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        self.sample_rng(rng)
    }
}

impl Strategy {
    #[inline]
    fn select(iter: Option<usize>, denom: Option<Outcome>) -> Strategy {
        if let Some(iter) = iter
            && iter < DIRECT_MAX_ITERATIONS
            && let Some(denom) = denom
        {
            Strategy::Direct(denom)
        } else {
            Strategy::Approx
        }
    }
}

#[inline]
fn gcd(mut m: u128, mut n: u128) -> u128 {
    // Use Stein's algorithm
    if m == 0 || n == 0 {
        return m | n;
    }

    // find common factors of 2
    let shift = (m | n).trailing_zeros();

    // divide n and m by 2 until odd
    m >>= m.trailing_zeros();
    n >>= n.trailing_zeros();

    while m != n {
        if m > n {
            m -= n;
            m >>= m.trailing_zeros();
        } else {
            n -= m;
            n >>= n.trailing_zeros();
        }
    }
    m << shift
}

#[inline]
fn fold_with<'a, I, T, F>(x: I, f: F) -> T
where
    I: IntoIterator<Item = &'a T>,
    T: Clone + 'a,
    F: Fn(&T, &T) -> T,
{
    let mut x = x.into_iter();
    let mut result = x.next().unwrap().clone();
    for i in x {
        result = f(&result, i);
    }
    result
}
