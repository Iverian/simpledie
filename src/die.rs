use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::iter::Zip;
use std::ops::Range;
use std::sync::Arc;
use std::{slice, vec};

use itertools::Itertools;
use rand::{rng, Rng, RngCore};

use crate::value::{ComputableValue, DefaultValue, Value};

pub type Outcome = u128;
pub type Iter<'a, T> = Zip<slice::Iter<'a, T>, slice::Iter<'a, Outcome>>;

type Map<T> = BTreeMap<T, Outcome>;
type Ptr<T> = Arc<T>;

#[derive(Clone)]
pub struct Die<T = DefaultValue>(Ptr<DieInner<T>>)
where
    T: Value;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DieInner<T = DefaultValue>
where
    T: Value,
{
    denom: Outcome,
    outcomes: Vec<Outcome>,
    values: Vec<T>,
}

trait DieLike<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome;
    fn values(&self) -> &[T];
    fn outcomes(&self) -> &[Outcome];
}

impl Die {
    #[must_use]
    pub fn numeric(value: DefaultValue) -> Self {
        Die::uniform(1..=value)
    }
}

impl<T> Die<T>
where
    T: Value,
{
    #[must_use]
    pub fn zero() -> Self {
        Self(Ptr::new(DieInner {
            values: Vec::new(),
            outcomes: Vec::new(),
            denom: 0,
        }))
    }

    #[must_use]
    pub fn scalar(value: T) -> Self {
        Self(Ptr::new(DieInner {
            values: vec![value],
            outcomes: vec![1],
            denom: 1,
        }))
    }

    #[must_use]
    pub fn uniform<I>(values: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let values: Vec<_> = values.into_iter().collect();
        let n = values.len();
        Self(Ptr::new(DieInner {
            values,
            outcomes: vec![1; n],
            denom: n as Outcome,
        }))
    }

    #[must_use]
    pub fn denom(&self) -> Outcome {
        self.0.denom
    }

    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.0.values
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.0.outcomes
    }

    #[must_use]
    pub fn min_value(&self) -> &T {
        self.0.values.first().unwrap()
    }

    #[must_use]
    pub fn max_value(&self) -> &T {
        self.0.values.last().unwrap()
    }

    #[must_use]
    pub fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        let x = rng.random_range(0..self.0.denom);
        let mut pos = 0;
        for (v, o) in self.0.values.iter().zip(&self.0.outcomes) {
            pos += o;
            if x < pos {
                return v;
            }
        }
        unreachable!()
    }

    #[must_use]
    pub fn sample_many_rng<G>(&self, n: usize, rng: &mut G) -> Vec<&T>
    where
        G: RngCore,
    {
        (0..n).map(|_| self.sample_rng(rng)).collect()
    }

    #[must_use]
    pub fn sample(&self) -> &T {
        let mut rng = rng();
        self.sample_rng(&mut rng)
    }

    #[must_use]
    pub fn sample_many(&self, n: usize) -> Vec<&T> {
        let mut rng = rng();
        self.sample_many_rng(n, &mut rng)
    }

    #[must_use]
    pub fn modes(&self) -> Vec<&T> {
        self.0.modes()
    }

    #[must_use]
    pub fn mode(&self) -> Option<&T> {
        self.0.mode()
    }

    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.0.probabilities()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.0.values.iter().zip(self.0.outcomes.iter())
    }

    #[must_use]
    pub fn map<O, F>(&self, f: F) -> Die<O>
    where
        O: Value,
        F: Fn(&T) -> O,
    {
        Die(Ptr::new(self.0.map(f)))
    }

    #[must_use]
    pub fn apply_two<R, O, F>(&self, rhs: &Die<R>, f: F) -> Die<O>
    where
        R: Value,
        O: Value,
        F: Fn(&T, &R) -> O,
    {
        Die(Ptr::new(self.0.apply_two(&rhs.0, f)))
    }

    #[must_use]
    pub fn apply_three<M, R, O, F>(&self, mhs: &Die<M>, rhs: &Die<R>, f: F) -> Die<O>
    where
        M: Value,
        R: Value,
        O: Value,
        F: Fn(&T, &M, &R) -> O,
    {
        Die(Ptr::new(self.0.apply_three(&mhs.0, &rhs.0, f)))
    }

    #[must_use]
    pub fn apply<I, D, O, F>(dice: I, f: F) -> Die<O>
    where
        I: Borrow<[D]>,
        D: Borrow<Self>,
        O: Value,
        F: Fn(&[&T]) -> O,
    {
        Die(Ptr::new(DieInner::apply(dice, f)))
    }

    #[must_use]
    pub fn repeat(&self, n: usize) -> Die<Vec<T>> {
        let dice = vec![self; n];
        Die::combine(dice)
    }

    #[must_use]
    pub fn combine<I, D>(dice: I) -> Die<Vec<T>>
    where
        I: Borrow<[D]>,
        D: Borrow<Self>,
    {
        Die(Ptr::new(DieInner::combine(dice)))
    }

    #[must_use]
    pub fn fold<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T + Clone + Copy,
    {
        if n == 0 {
            return Self::zero();
        }
        if n == 1 {
            return self.clone();
        }

        Die(Ptr::new(self.0.fold(n, f)))
    }

    #[must_use]
    pub fn fold_assoc<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T + Clone + Copy,
    {
        if n == 0 {
            return Self::zero();
        }
        if n == 1 {
            return self.clone();
        }
        if n == 2 {
            return self.apply_two(self, f);
        }

        Die(Ptr::new(self.0.fold_assoc(n, f)))
    }

    #[must_use]
    pub fn explode<P, F>(&self, range: Range<usize>, p: P, f: F) -> Self
    where
        P: Fn(&[T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        if range.end == 0 {
            return Self::zero();
        }
        if range.end == 1 {
            return self.clone();
        }

        Self(Ptr::new(self.0.explode(range.start, range.end, p, f)))
    }

    #[must_use]
    pub(crate) fn from_map(map: Map<T>, denom: Outcome) -> Self {
        Self(Ptr::new(DieInner::from_map(map, denom)))
    }
}

impl<T> Die<T>
where
    T: ComputableValue,
{
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.0.mean()
    }

    #[must_use]
    pub fn variance(&self) -> f64 {
        self.0.variance()
    }

    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.0.stddev()
    }
}

impl<T> DieLike<T> for Die<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome {
        self.0.denom
    }

    fn values(&self) -> &[T] {
        &self.0.values
    }

    fn outcomes(&self) -> &[Outcome] {
        &self.0.outcomes
    }
}

impl<T> DieLike<T> for &Die<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome {
        self.0.denom
    }

    fn values(&self) -> &[T] {
        &self.0.values
    }

    fn outcomes(&self) -> &[Outcome] {
        &self.0.outcomes
    }
}

impl<T> Debug for Die<T>
where
    T: Value,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Die")
            .field("denom", &self.0.denom)
            .field("values", &self.0.values)
            .field("outcomes", &self.0.outcomes)
            .finish()
    }
}

impl<T> DieInner<T>
where
    T: Value,
{
    fn iter(&self) -> Iter<'_, T> {
        self.values.iter().zip(self.outcomes.iter())
    }

    #[must_use]
    fn modes(&self) -> Vec<&T> {
        self.values
            .iter()
            .zip(&self.outcomes)
            .max_set_by_key(|(_, o)| **o)
            .into_iter()
            .map(|(v, _)| v)
            .collect()
    }

    #[must_use]
    fn mode(&self) -> Option<&T> {
        self.values
            .iter()
            .zip(&self.outcomes)
            .max_by_key(|(_, o)| **o)
            .map(|(v, _)| v)
    }

    #[must_use]
    fn probabilities(&self) -> Vec<f64> {
        self.outcomes
            .iter()
            .map(|x| *x as f64 / self.denom as f64)
            .collect_vec()
    }

    #[must_use]
    fn from_map(map: Map<T>, mut denom: Outcome) -> Self {
        let mut values = Vec::with_capacity(map.len());
        let mut outcomes = Vec::with_capacity(map.len());
        let mut acc = denom;
        for (value, outcome) in map {
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
    fn map<O, F>(&self, f: F) -> DieInner<O>
    where
        O: Value,
        F: Fn(&T) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            let o = map.entry(f(v1)).or_default();
            *o += o1;
        }

        DieInner::from_map(map, self.denom)
    }

    #[must_use]
    fn apply_two<R, O, F>(&self, rhs: &DieInner<R>, f: F) -> DieInner<O>
    where
        R: Value,
        O: Value,
        F: Fn(&T, &R) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            for (v2, o2) in rhs.iter() {
                let o = map.entry(f(v1, v2)).or_default();
                *o += o1 * o2;
            }
        }

        DieInner::from_map(map, self.denom * rhs.denom)
    }

    #[must_use]
    fn apply_three<M, R, O, F>(&self, mhs: &DieInner<M>, rhs: &DieInner<R>, f: F) -> DieInner<O>
    where
        M: Value,
        R: Value,
        O: Value,
        F: Fn(&T, &M, &R) -> O,
    {
        let mut map = Map::new();

        for (v1, o1) in self.iter() {
            for (v2, o2) in mhs.iter() {
                for (v3, o3) in rhs.iter() {
                    let o = map.entry(f(v1, v2, v3)).or_default();
                    *o += o1 * o2 * o3;
                }
            }
        }

        DieInner::from_map(map, self.denom * rhs.denom)
    }

    #[must_use]
    fn apply<I, Q, D, O, F>(dice: I, f: F) -> DieInner<O>
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

        DieInner::from_map(
            map,
            dice.iter().fold(1 as Outcome, |mut acc, x| {
                acc *= x.borrow().denom();
                acc
            }),
        )
    }

    #[must_use]
    fn combine<I, Q, D>(dice: I) -> DieInner<Vec<T>>
    where
        I: Borrow<[Q]>,
        Q: Borrow<D>,
        D: DieLike<T>,
    {
        let dice = dice.borrow();
        let total_len = Self::iterations_of(&dice).expect("iteration overflow");
        let denom = Self::denom_of(&dice).expect("denominator overflow");
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

        DieInner {
            denom,
            outcomes,
            values,
        }
    }

    #[must_use]
    fn fold<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T + Clone + Copy,
    {
        let mut die = self.apply_two(self, f);
        for _ in 0..n.saturating_sub(2) {
            die = die.apply_two(self, f);
        }

        die
    }

    #[must_use]
    fn fold_assoc<F>(&self, n: usize, f: F) -> Self
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
    fn explode<P, F>(&self, min: usize, max: usize, p: P, f: F) -> Self
    where
        P: Fn(&[T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        fn fold_with<T, F>(x: &[T], f: F) -> T
        where
            T: Clone,
            F: Fn(&T, &T) -> T,
        {
            let mut result = x[0].clone();
            for i in x.iter().skip(1) {
                result = f(&result, i);
            }
            result
        }

        let m = self.values.len();
        let mut map = Map::new();
        let mut denom = 0;

        let dice = Self::combine::<_, _, Self>(vec![self; min]);
        let mut items = dice
            .values
            .into_iter()
            .zip(dice.outcomes)
            .map(|(v, o)| (fold_with(&v, &f), v, o))
            .collect_vec();

        for _ in min..max {
            let mut new_items = Vec::with_capacity(items.len() * m);

            for (r1, v1, o1) in items {
                let mut negative_outcome = 0;

                for (v2, o2) in self.iter() {
                    let outcome = o1 * *o2;
                    let mut value = v1.clone();
                    value.push(v2.clone());
                    if p(&value) {
                        new_items.push((f(&r1, v2), value, outcome));
                    } else {
                        negative_outcome += outcome;
                    }
                }

                if negative_outcome != 0 {
                    let o = map.entry(r1).or_insert(0);
                    *o += negative_outcome;
                    denom += negative_outcome;
                }
            }

            items = new_items;
        }

        for (v1, _, o1) in items {
            let o = map.entry(v1).or_insert(0);
            *o += o1;
            denom += o1;
        }

        Self::from_map(map, denom)
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
    fn mean(&self) -> f64 {
        self.mean_computed(self.values.iter().map(|x| x.compute()))
    }

    #[must_use]
    fn variance(&self) -> f64 {
        let computed = self.compute();
        let mean = self.mean_computed(computed.iter());

        computed
            .iter()
            .zip(&self.outcomes)
            .fold(0.0, |mut acc, (value, outcome)| {
                acc += (*value - mean).powi(2) * (*outcome as f64);
                acc
            })
            / (self.denom as f64)
    }

    #[must_use]
    fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn compute(&self) -> Vec<f64> {
        self.values.iter().map(|x| x.compute()).collect()
    }

    fn mean_computed<I, F>(&self, computed: I) -> f64
    where
        I: Iterator<Item = F>,
        F: Borrow<f64>,
    {
        computed
            .zip(&self.outcomes)
            .fold(0.0, |mut acc, (value, outcome)| {
                acc += value.borrow() * (*outcome as f64);
                acc
            })
            / (self.denom as f64)
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
}

impl<'a, T> IntoIterator for &'a Die<T>
where
    T: Value + 'static,
{
    type Item = (&'a T, &'a Outcome);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
