use std::ops::{Add, Mul, Sub};

use crate::Die;

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
