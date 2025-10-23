use std::borrow::Borrow;
use std::fmt::Debug;
use std::iter::Zip;
use std::{slice, vec};

use rand::{rng, RngCore};

use super::inner::DieInner;
use super::DieLike;
use crate::value::{ComputableValue, ComputedValue, DefaultValue, Value};
use crate::{Outcome, Ptr, Result, MAX_EXPLODE, MIN_EXPLODE};

pub type Iter<'a, T> = Zip<slice::Iter<'a, T>, slice::Iter<'a, Outcome>>;

#[derive(Clone)]
pub struct Die<T = DefaultValue>(Ptr<DieInner<T>>)
where
    T: Value;

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
    pub fn scalar(value: T) -> Self {
        Self::new(DieInner::scalar(value))
    }

    #[must_use]
    pub fn uniform<I>(values: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::new(DieInner::uniform(values))
    }

    #[must_use]
    pub fn denom(&self) -> Outcome {
        self.0.denom()
    }

    #[must_use]
    pub fn values(&self) -> &[T] {
        self.0.values()
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        self.0.outcomes()
    }

    #[must_use]
    pub fn min_value(&self) -> &T {
        self.0.values().first().unwrap()
    }

    #[must_use]
    pub fn max_value(&self) -> &T {
        self.0.values().last().unwrap()
    }

    #[must_use]
    pub fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        self.0.sample_rng(rng)
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
        self.0.iter()
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
    pub fn apply_three<T2, T3, O, F>(&self, d2: &Die<T2>, d3: &Die<T3>, f: F) -> Die<O>
    where
        T2: Value,
        T3: Value,
        O: Value,
        F: Fn(&T, &T2, &T3) -> O,
    {
        Die(Ptr::new(self.0.apply_three(&d2.0, &d3.0, f)))
    }

    #[must_use]
    pub fn apply_four<T2, T3, T4, O, F>(
        &self,
        d2: &Die<T2>,
        d3: &Die<T3>,
        d4: &Die<T4>,
        f: F,
    ) -> Die<O>
    where
        T2: Value,
        T3: Value,
        T4: Value,
        O: Value,
        F: Fn(&T, &T2, &T3, &T4) -> O,
    {
        Die::new(self.0.apply_four(&d2.0, &d3.0, &d4.0, f))
    }

    #[must_use]
    pub fn apply<I, D, O, F>(dice: I, f: F) -> Die<O>
    where
        I: Borrow<[D]>,
        D: Borrow<Self>,
        O: Value,
        F: Fn(&[&T]) -> O,
    {
        Die::new(DieInner::apply(dice, f))
    }

    pub fn repeat(&self, n: usize) -> Result<Die<Vec<T>>> {
        let dice = vec![self; n];
        Die::combine(dice)
    }

    pub fn combine<I, D>(dice: I) -> Result<Die<Vec<T>>>
    where
        I: Borrow<[D]>,
        D: Borrow<Self>,
    {
        Ok(Die::new(DieInner::combine(dice)?))
    }

    #[must_use]
    pub fn fold<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        if n == 0 {
            panic!("fold number cannot be zero");
        }
        if n == 1 {
            return self.clone();
        }

        Die::new(self.0.fold(n, f))
    }

    #[must_use]
    pub fn fold_assoc<F>(&self, n: usize, f: F) -> Self
    where
        F: Fn(&T, &T) -> T + Clone + Copy,
    {
        if n == 0 {
            panic!("fold number cannot be zero");
        }
        if n == 1 {
            return self.clone();
        }
        if n == 2 {
            return self.apply_two(self, f);
        }

        Self::new(self.0.fold_assoc(n, f))
    }

    #[must_use]
    pub fn explode<P, F>(&self, min: usize, max: usize, predicate: P, fold: F) -> Self
    where
        P: Fn(&[&T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        if !(MIN_EXPLODE..=MAX_EXPLODE).contains(&min) {
            panic!("min explode out of range");
        }
        if !(MIN_EXPLODE..=MAX_EXPLODE).contains(&max) {
            panic!("max explode out of range");
        }
        if max == 1 {
            return self.clone();
        }

        Self::new(self.0.explode(min.clamp(1, max), max, predicate, fold))
    }

    #[must_use]
    pub fn explode_one<P, F>(&self, max: usize, predicate: P, fold: F) -> Self
    where
        P: Fn(&[&T]) -> bool,
        F: Fn(&T, &T) -> T,
    {
        self.explode(1, max, predicate, fold)
    }

    #[must_use]
    pub(crate) fn new(value: DieInner<T>) -> Self {
        Self(Ptr::new(value))
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

    #[must_use]
    pub fn computed_values(&self) -> Vec<ComputedValue> {
        self.0.computed_values()
    }
}

impl<T> DieLike<T> for Die<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome {
        self.0.denom()
    }

    fn values(&self) -> &[T] {
        self.0.values()
    }

    fn outcomes(&self) -> &[Outcome] {
        self.0.outcomes()
    }

    fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        self.0.sample_rng(rng)
    }
}

impl<T> DieLike<T> for &Die<T>
where
    T: Value,
{
    fn denom(&self) -> Outcome {
        self.0.denom()
    }

    fn values(&self) -> &[T] {
        self.0.values()
    }

    fn outcomes(&self) -> &[Outcome] {
        self.0.outcomes()
    }

    fn sample_rng<G>(&self, rng: &mut G) -> &T
    where
        G: RngCore,
    {
        self.0.sample_rng(rng)
    }
}

impl<T> Debug for Die<T>
where
    T: Value,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Die")
            .field("denom", &self.0.denom())
            .field("values", &self.0.values())
            .field("outcomes", &self.0.outcomes())
            .finish()
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
