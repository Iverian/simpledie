use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use itertools::Itertools;

use crate::{Die, Value};

impl<L> Die<L>
where
    L: Value,
{
    #[must_use]
    fn vadd<R>(self, rhs: R) -> Die<L::Output>
    where
        R: Value,
        R: Value,
        L: Add<R>,
        L::Output: Value,
    {
        self.map(|x| x.clone().add(rhs.clone()))
    }
}

impl<L> Die<L>
where
    L: Value,
{
    #[must_use]
    fn vsub<R>(self, rhs: R) -> Die<L::Output>
    where
        R: Value,
        R: Value,
        L: Sub<R>,
        L::Output: Value,
    {
        self.map(|x| x.clone().sub(rhs.clone()))
    }
}

impl<L> Die<L>
where
    L: Value,
{
    #[must_use]
    fn add<R>(self, rhs: Die<R>) -> Die<L::Output>
    where
        R: Value,
        R: Value,
        L: Add<R>,
        L::Output: Value,
    {
        self.apply_two(&rhs, |x, y| x.clone().add(y.clone()))
    }
}

impl<L> Die<L>
where
    L: Value,
{
    #[must_use]
    fn sub<R>(self, rhs: Die<R>) -> Die<L::Output>
    where
        R: Value,
        R: Value,
        L: Sub<R>,
        L::Output: Value,
    {
        self.apply_two(&rhs, |x, y| x.clone().sub(y.clone()))
    }
}

impl<T> Die<T>
where
    T: Value,
    T: Add<T, Output = T>,
{
    #[must_use]
    pub fn nsum(&self, n: usize) -> Die<T> {
        self.fold_assoc(n, |x, y| x.clone().add(y.clone()))
    }
}

impl<T> Die<T>
where
    T: Value,
{
    #[must_use]
    pub fn compare(&self, value: T) -> Die<Ordering> {
        self.map(|x| x.cmp(&value))
    }

    #[must_use]
    pub fn eq(&self, value: T) -> Die<bool> {
        self.map(|x| x == &value)
    }

    #[must_use]
    pub fn neq(&self, value: T) -> Die<bool> {
        self.map(|x| x != &value)
    }

    #[must_use]
    pub fn contains(&self, values: &[T]) -> Die<bool> {
        self.map(|x| values.contains(x))
    }

    #[must_use]
    pub fn lt(&self, value: T) -> Die<bool> {
        self.map(|x| x < &value)
    }

    #[must_use]
    pub fn le(&self, value: T) -> Die<bool> {
        self.map(|x| x <= &value)
    }

    #[must_use]
    pub fn gt(&self, value: T) -> Die<bool> {
        self.map(|x| x > &value)
    }

    #[must_use]
    pub fn ge(&self, value: T) -> Die<bool> {
        self.map(|x| x >= &value)
    }

    #[must_use]
    pub fn nmax(&self, n: usize) -> Die<T> {
        self.fold_assoc(n, |x, y| x.clone().max(y.clone()))
    }

    #[must_use]
    pub fn nmin(&self, n: usize) -> Die<T> {
        self.fold_assoc(n, |x, y| x.clone().min(y.clone()))
    }

    #[must_use]
    pub fn lowest(&self, total: usize, keep: usize) -> Die<Vec<T>> {
        let dice = vec![self; total];
        Die::apply(dice, |x| {
            x.iter()
                .sorted()
                .take(keep)
                .map(|&x| x.clone())
                .collect_vec()
        })
    }

    #[must_use]
    pub fn highest(&self, total: usize, keep: usize) -> Die<Vec<T>> {
        let dice = vec![self; total];
        let skip = total.saturating_sub(keep);
        Die::apply(dice, |x| {
            x.iter()
                .sorted()
                .skip(skip)
                .map(|&x| x.clone())
                .collect_vec()
        })
    }
}

impl Die<bool> {
    #[must_use]
    pub fn neg(&self) -> Self {
        self.map(|x| !x)
    }

    #[must_use]
    pub fn branch<T>(&self, lhs: &Die<T>, rhs: &Die<T>) -> Die<T>
    where
        T: Value,
    {
        self.apply_three(lhs, rhs, |x, y, z| if *x { y.clone() } else { z.clone() })
    }
}

impl<T> Die<Vec<T>>
where
    T: Value,
{
    #[must_use]
    pub fn max(&self) -> Die<T> {
        self.map(|x| x.iter().max().expect("non empty dice").clone())
    }

    #[must_use]
    pub fn min(&self) -> Die<T> {
        self.map(|x| x.iter().min().expect("non empty dice").clone())
    }
}

impl<T> Die<Vec<T>>
where
    T: Value,
    T: Sum<T>,
{
    #[must_use]
    pub fn sum(&self) -> Die<T> {
        self.map(|x| x.iter().cloned().sum())
    }
}

impl<L, R> Add<R> for Die<L>
where
    L: Value,
    R: Value,
    L: Add<R>,
    L::Output: Value,
{
    type Output = Die<L::Output>;

    fn add(self, rhs: R) -> Self::Output {
        self.vadd(rhs)
    }
}

impl<L, R> Sub<R> for Die<L>
where
    L: Value,
    R: Value,
    L: Sub<R>,
    L::Output: Value,
{
    type Output = Die<L::Output>;

    fn sub(self, rhs: R) -> Self::Output {
        self.vsub(rhs)
    }
}

impl<L, R> Add<Die<R>> for Die<L>
where
    L: Value,
    R: Value,
    L: Add<R>,
    L::Output: Value,
{
    type Output = Die<L::Output>;

    fn add(self, rhs: Die<R>) -> Self::Output {
        self.add(rhs)
    }
}

impl<L, R> Sub<Die<R>> for Die<L>
where
    L: Value,
    R: Value,
    L: Sub<R>,
    L::Output: Value,
{
    type Output = Die<L::Output>;

    fn sub(self, rhs: Die<R>) -> Self::Output {
        self.sub(rhs)
    }
}

impl<T> Mul<Die<T>> for usize
where
    T: Value,
    T: Add<T, Output = T>,
{
    type Output = Die<T>;

    fn mul(self, rhs: Die<T>) -> Self::Output {
        rhs.nsum(self)
    }
}
