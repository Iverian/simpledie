pub mod composite;
pub mod ext;

use std::cmp::Ordering;
use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::util::DefaultKey;
use crate::Key;

pub trait RawOperation<K = DefaultKey>: Debug + DynClone
where
    K: Key,
{
    type Output: Key;

    fn call(&self, values: &[K]) -> Self::Output;

    fn shift_indices(&mut self, value: usize);

    fn boxed(self) -> Boxed<K, Self::Output>
    where
        Self: Sized + 'static,
    {
        Boxed(Box::new(self))
    }
}

pub trait Operation<K = DefaultKey>: RawOperation<K> + Clone
where
    K: Key,
{
}

impl<K, T> Operation<K> for T
where
    K: Key,
    T: RawOperation<K> + Clone,
{
}

impl<K, O> Clone for Box<dyn RawOperation<K, Output = O>>
where
    K: Key,
    O: Key,
{
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Id(usize);

#[derive(Clone, Copy)]
pub struct Map<T, F>(T, F);

#[derive(Clone, Copy, Debug)]
pub struct Neg<T>(T);

#[derive(Clone, Copy, Debug)]
pub struct AddKey<T, R>(T, R);

#[derive(Clone, Copy, Debug)]
pub struct SubKey<T, R>(T, R);

#[derive(Clone, Copy, Debug)]
pub struct MulKey<T, R>(T, R);

#[derive(Clone, Copy, Debug)]
pub struct DivKey<T, R>(T, R);

#[derive(Clone, Copy, Debug)]
pub struct Not<T>(T);

#[derive(Clone, Debug)]
pub struct Eq<T, R, const N: usize = 1>(T, [R; N]);

#[derive(Clone, Copy, Debug)]
pub struct Cmp<T, R>(T, R);

#[derive(Clone, Copy, Debug)]
pub struct Add<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct Sub<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct Mul<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct Div<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct Min<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct Max<L, R>(L, R);

#[derive(Clone)]
pub struct Fold<T, F>(Vec<T>, F);

#[derive(Clone, Copy)]
pub struct FoldTwo<T1, T2, F>(T1, T2, F);

#[derive(Clone, Copy)]
pub struct FoldThree<T1, T2, T3, F>(T1, T2, T3, F);

#[derive(Clone, Copy)]
pub struct FoldFour<T1, T2, T3, T4, F>(T1, T2, T3, T4, F);

#[derive(Clone, Copy)]
pub struct FoldFive<T1, T2, T3, T4, T5, F>(T1, T2, T3, T4, T5, F);

#[derive(Clone, Debug)]
pub struct Sum<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct Product<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct MaxOf<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct MinOf<T>(Vec<T>);

#[derive(Clone)]
pub struct Any<T, F>(Vec<T>, F);

#[derive(Clone)]
pub struct All<T, F>(Vec<T>, F);

#[derive(Clone, Copy)]
pub struct Branch<F, C, L, R>(F, C, L, R);

#[derive(Clone, Debug)]
pub struct Boxed<K = DefaultKey, O = DefaultKey>(Box<dyn RawOperation<K, Output = O>>)
where
    K: Key,
    O: Key;

impl<K> RawOperation<K> for Id
where
    K: Key,
{
    type Output = K;

    fn call(&self, values: &[K]) -> Self::Output {
        values[self.0]
    }

    fn shift_indices(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<K, O, T, F> RawOperation<K> for Map<T, F>
where
    K: Key,
    O: Key,
    T: Operation<K>,
    F: Fn(T::Output) -> O + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.1(self.0.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T, F> Debug for Map<T, F>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Map").field(&self.0).finish()
    }
}

impl<K, O, T> RawOperation<K> for Neg<T>
where
    K: Key,
    O: Key + std::ops::Neg,
    O::Output: Key,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = O::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        -self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T> RawOperation<K> for AddKey<T, R>
where
    K: Key,
    O: Key + std::ops::Add<R>,
    O::Output: Key,
    R: Copy + Debug,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = O::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) + self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T> RawOperation<K> for SubKey<T, R>
where
    K: Key,
    O: Key + std::ops::Sub<R>,
    O::Output: Key,
    R: Copy + Debug,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = O::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) - self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T> RawOperation<K> for MulKey<T, R>
where
    K: Key,
    O: Key + std::ops::Mul<R>,
    O::Output: Key,
    R: Copy + Debug,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = O::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) * self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T> RawOperation<K> for DivKey<T, R>
where
    K: Key,
    O: Key + std::ops::Div<R>,
    O::Output: Key,
    R: Copy + Debug,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = O::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) / self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, T> RawOperation<K> for Not<T>
where
    K: Key,
    O: Key + Into<bool>,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        !self.0.call(values).into()
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T, const N: usize> RawOperation<K> for Eq<T, R, N>
where
    K: Key,
    R: Copy + Debug + Into<O>,
    O: Key,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        self.1.iter().any(|x| self.0.call(values) == (*x).into())
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O, R, T> RawOperation<K> for Cmp<T, R>
where
    K: Key,
    R: Copy + Debug + Into<O>,
    O: Key,
    T: RawOperation<K, Output = O> + Clone,
{
    type Output = Ordering;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).cmp(&self.1.into())
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, LO, RO, L, R> RawOperation<K> for Add<L, R>
where
    K: Key,
    LO: Key + std::ops::Add<RO>,
    RO: Key,
    LO::Output: Key,
    L: RawOperation<K, Output = LO> + Clone,
    R: RawOperation<K, Output = RO> + Clone,
{
    type Output = LO::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) + self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, LO, RO, L, R> RawOperation<K> for Sub<L, R>
where
    K: Key,
    LO: Key + std::ops::Sub<RO>,
    RO: Key,
    LO::Output: Key,
    L: RawOperation<K, Output = LO> + Clone,
    R: RawOperation<K, Output = RO> + Clone,
{
    type Output = LO::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) - self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, LO, RO, L, R> RawOperation<K> for Mul<L, R>
where
    K: Key,
    LO: Key + std::ops::Mul<RO>,
    RO: Key,
    LO::Output: Key,
    L: RawOperation<K, Output = LO> + Clone,
    R: RawOperation<K, Output = RO> + Clone,
{
    type Output = LO::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) * self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, LO, RO, L, R> RawOperation<K> for Div<L, R>
where
    K: Key,
    LO: Key + std::ops::Div<RO>,
    RO: Key,
    LO::Output: Key,
    L: RawOperation<K, Output = LO> + Clone,
    R: RawOperation<K, Output = RO> + Clone,
{
    type Output = LO::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) / self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, O, L, R> RawOperation<K> for Min<L, R>
where
    K: Key,
    O: Key,
    L: RawOperation<K, Output = O> + Clone,
    R: RawOperation<K, Output = O> + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).min(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, O, L, R> RawOperation<K> for Max<L, R>
where
    K: Key,
    O: Key,
    L: RawOperation<K, Output = O> + Clone,
    R: RawOperation<K, Output = O> + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).max(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, R, T, F> RawOperation<K> for Fold<T, F>
where
    K: Key,
    R: Key,
    T: Operation<K>,
    F: Fn(&[T::Output]) -> R + Clone,
{
    type Output = R;

    fn call(&self, values: &[K]) -> Self::Output {
        self.1(
            self.0
                .iter()
                .map(|x| x.call(values))
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T, F> Debug for Fold<T, F>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Fold").field(&self.0).finish()
    }
}

impl<K, R, T1, T2, F> RawOperation<K> for FoldTwo<T1, T2, F>
where
    K: Key,
    R: Key,
    T1: Operation<K>,
    T2: Operation<K>,
    F: Fn(T1::Output, T2::Output) -> R + Clone,
{
    type Output = R;

    fn call(&self, values: &[K]) -> Self::Output {
        self.2(self.0.call(values), self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<T1, T2, F> Debug for FoldTwo<T1, T2, F>
where
    T1: Debug,
    T2: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FoldTwo")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<K, R, T1, T2, T3, F> RawOperation<K> for FoldThree<T1, T2, T3, F>
where
    K: Key,
    R: Key,
    T1: Operation<K>,
    T2: Operation<K>,
    T3: Operation<K>,
    F: Fn(T1::Output, T2::Output, T3::Output) -> R + Clone,
{
    type Output = R;

    fn call(&self, values: &[K]) -> R {
        self.3(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
        )
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
        self.2.shift_indices(value);
    }
}

impl<T1, T2, T3, F> Debug for FoldThree<T1, T2, T3, F>
where
    T1: Debug,
    T2: Debug,
    T3: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FoldThree")
            .field(&self.0)
            .field(&self.1)
            .field(&self.2)
            .finish()
    }
}

impl<K, R, T1, T2, T3, T4, F> RawOperation<K> for FoldFour<T1, T2, T3, T4, F>
where
    K: Key,
    R: Key,
    T1: Operation<K>,
    T2: Operation<K>,
    T3: Operation<K>,
    T4: Operation<K>,
    F: Fn(T1::Output, T2::Output, T3::Output, T4::Output) -> R + Clone,
{
    type Output = R;

    fn call(&self, values: &[K]) -> Self::Output {
        self.4(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
            self.3.call(values),
        )
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
        self.2.shift_indices(value);
        self.3.shift_indices(value);
    }
}

impl<T1, T2, T3, T4, F> Debug for FoldFour<T1, T2, T3, T4, F>
where
    T1: Debug,
    T2: Debug,
    T3: Debug,
    T4: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FoldFour")
            .field(&self.0)
            .field(&self.1)
            .field(&self.2)
            .field(&self.3)
            .finish()
    }
}

impl<K, R, T1, T2, T3, T4, T5, F> RawOperation<K> for FoldFive<T1, T2, T3, T4, T5, F>
where
    K: Key,
    R: Key,
    T1: Operation<K>,
    T2: Operation<K>,
    T3: Operation<K>,
    T4: Operation<K>,
    T5: Operation<K>,
    F: Fn(T1::Output, T2::Output, T3::Output, T4::Output, T5::Output) -> R + Clone,
{
    type Output = R;

    fn call(&self, values: &[K]) -> Self::Output {
        self.5(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
            self.3.call(values),
            self.4.call(values),
        )
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
        self.2.shift_indices(value);
        self.3.shift_indices(value);
        self.4.shift_indices(value);
    }
}

impl<T1, T2, T3, T4, T5, F> Debug for FoldFive<T1, T2, T3, T4, T5, F>
where
    T1: Debug,
    T2: Debug,
    T3: Debug,
    T4: Debug,
    T5: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FoldFive")
            .field(&self.0)
            .field(&self.1)
            .field(&self.2)
            .field(&self.3)
            .field(&self.4)
            .finish()
    }
}

impl<K, O, T> RawOperation<K> for Sum<T>
where
    K: Key,
    O: Key + std::iter::Sum,
    T: Operation<K, Output = O>,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).sum()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, O, T> RawOperation<K> for Product<T>
where
    K: Key,
    O: Key + std::iter::Product,
    T: Operation<K, Output = O>,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).product()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, O, T> RawOperation<K> for MinOf<T>
where
    K: Key,
    O: Key,
    T: Operation<K, Output = O>,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).min().unwrap()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, O, T> RawOperation<K> for MaxOf<T>
where
    K: Key,
    O: Key,
    T: Operation<K, Output = O>,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).max().unwrap()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, O, T, F> RawOperation<K> for Any<T, F>
where
    K: Key,
    T: Operation<K, Output = O>,
    F: Fn(O) -> bool + Clone,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().any(|x| self.1(x.call(values)))
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T, F> Debug for Any<T, F>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Any").field(&self.0).finish()
    }
}

impl<K, O, T, F> RawOperation<K> for All<T, F>
where
    K: Key,
    O: Key,
    T: Operation<K, Output = O>,
    F: Fn(O) -> bool + Clone,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().all(|x| self.1(x.call(values)))
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T, F> Debug for All<T, F>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("All").field(&self.0).finish()
    }
}

impl<K, O, C, L, R, F> RawOperation<K> for Branch<F, C, L, R>
where
    K: Key,
    O: Key,
    C: Operation<K>,
    L: Operation<K, Output = O>,
    R: Operation<K, Output = O>,
    F: Fn(C::Output) -> bool + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        if self.0(self.1.call(values)) {
            self.2.call(values)
        } else {
            self.3.call(values)
        }
    }

    fn shift_indices(&mut self, value: usize) {
        self.1.shift_indices(value);
        self.2.shift_indices(value);
        self.3.shift_indices(value);
    }
}

impl<F, C, L, R> Debug for Branch<F, C, L, R>
where
    C: Debug,
    L: Debug,
    R: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Branch")
            .field(&self.1)
            .field(&self.2)
            .field(&self.3)
            .finish()
    }
}

impl<K, O> RawOperation<K> for Boxed<K, O>
where
    K: Key,
    O: Key,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}
