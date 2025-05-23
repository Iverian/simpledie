pub mod composite;
pub mod ext;

use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::util::Key;

pub trait Operation<K>: Debug + DynClone
where
    K: Clone + Copy + Ord + Debug + Send,
{
    type Output: Clone + Copy + Ord + Debug + Send;

    fn call(&self, values: &[K]) -> Self::Output;

    fn shift_indices(&mut self, value: usize);

    fn boxed(self) -> Boxed<K, Self::Output>
    where
        Self: Sized + Send + 'static,
    {
        Boxed(Box::new(self))
    }
}

impl<K, O> Clone for Box<dyn Operation<K, Output = O>>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
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

#[derive(Clone)]
pub struct Boxed<K, O>(Box<dyn Operation<K, Output = O> + 'static>)
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord;

impl<K> Operation<K> for Id
where
    K: Clone + Copy + Ord + Debug + Send,
{
    type Output = K;

    fn call(&self, values: &[K]) -> Self::Output {
        values[self.0]
    }

    fn shift_indices(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<K, T, F, O> Operation<K> for Map<T, F>
where
    K: Clone + Copy + Ord + Debug + Send,
    O: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
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

impl<K, T> Operation<K> for Neg<T>
where
    K: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
    T::Output: std::ops::Neg,
    <T::Output as std::ops::Neg>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <T::Output as std::ops::Neg>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        -self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T, R> Operation<K> for AddKey<T, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    R: Clone + Copy + Debug,
    T: Operation<K> + Clone,
    T::Output: std::ops::Add<R>,
    <T::Output as std::ops::Add<R>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <T::Output as std::ops::Add<R>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) + self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T, R> Operation<K> for MulKey<T, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    R: Clone + Copy + Debug,
    T: Operation<K> + Clone,
    T::Output: std::ops::Mul<R>,
    <T::Output as std::ops::Mul<R>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <T::Output as std::ops::Mul<R>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) * self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T, R> Operation<K> for DivKey<T, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    R: Clone + Copy + Debug,
    T: Operation<K> + Clone,
    T::Output: std::ops::Div<R>,
    <T::Output as std::ops::Div<R>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <T::Output as std::ops::Div<R>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) / self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T> Operation<K> for Not<T>
where
    K: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
    T::Output: Into<bool>,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        !self.0.call(values).into()
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T, const N: usize> Operation<K> for Eq<T, T::Output, N>
where
    K: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
    T::Output: std::cmp::Eq + Debug,
{
    type Output = bool;

    fn call(&self, values: &[K]) -> Self::Output {
        self.1.contains(&self.0.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, T> Operation<K> for Cmp<T, T::Output>
where
    K: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
    T::Output: Debug,
{
    type Output = std::cmp::Ordering;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).cmp(&self.1)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, L, R> Operation<K> for Add<L, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    L: Operation<K> + Clone,
    R: Operation<K> + Clone,
    L::Output: std::ops::Add<R::Output>,
    <L::Output as std::ops::Add<R::Output>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <L::Output as std::ops::Add<R::Output>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) + self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, L, R> Operation<K> for Mul<L, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    L: Operation<K> + Clone,
    R: Operation<K> + Clone,
    L::Output: std::ops::Mul<R::Output>,
    <L::Output as std::ops::Mul<R::Output>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <L::Output as std::ops::Mul<R::Output>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) * self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, L, R> Operation<K> for Div<L, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    L: Operation<K> + Clone,
    R: Operation<K> + Clone,
    L::Output: std::ops::Div<R::Output>,
    <L::Output as std::ops::Div<R::Output>>::Output: Clone + Copy + Ord + Debug + Send,
{
    type Output = <L::Output as std::ops::Div<R::Output>>::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values) / self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, L, R> Operation<K> for Min<L, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    L: Operation<K, Output = K> + Clone,
    R: Operation<K, Output = K> + Clone,
{
    type Output = K;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).min(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, L, R> Operation<K> for Max<L, R>
where
    K: Clone + Copy + Ord + Debug + Send,
    L: Operation<K, Output = K> + Clone,
    R: Operation<K, Output = K> + Clone,
{
    type Output = K;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values).max(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<K, T, F, O> Operation<K> for Fold<T, F>
where
    K: Clone + Copy + Ord + Debug + Send,
    O: Clone + Copy + Ord + Debug + Send,
    T: Operation<K> + Clone,
    F: Fn(&[T::Output]) -> O + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.1(
            self.0
                .iter()
                .map(|x| x.call(values))
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .into()
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

impl<K, T1, T2, F, O> Operation<K> for FoldTwo<T1, T2, F>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
    T1: Operation<K> + Clone,
    T2: Operation<K> + Clone,
    F: Fn(T1::Output, T2::Output) -> O + Clone,
{
    type Output = O;

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

impl<K, T1, T2, T3, F, O> Operation<K> for FoldThree<T1, T2, T3, F>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
    T1: Operation<K> + Clone,
    T2: Operation<K> + Clone,
    T3: Operation<K> + Clone,
    F: Fn(T1::Output, T2::Output, T3::Output) -> O + Clone,
    O: Into<Key>,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
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

impl<K, T1, T2, T3, T4, F, O> Operation<K> for FoldFour<T1, T2, T3, T4, F>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
    T1: Operation<K> + Clone,
    T2: Operation<K> + Clone,
    T3: Operation<K> + Clone,
    T4: Operation<K> + Clone,
    F: Fn(T1::Output, T2::Output, T3::Output, T4::Output) -> O + Clone,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.4(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
            self.3.call(values),
        )
        .into()
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

impl<K, T1, T2, T3, T4, T5, F, O> Operation<K> for FoldFive<T1, T2, T3, T4, T5, F>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
    T1: Operation<K> + Clone,
    T2: Operation<K> + Clone,
    T3: Operation<K> + Clone,
    T4: Operation<K> + Clone,
    T5: Operation<K> + Clone,
    F: Fn(T1::Output, T2::Output, T3::Output, T4::Output, T5::Output) -> O + Clone,
    O: Into<Key>,
{
    type Output = O;

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

impl<K, T> Operation<K> for Sum<T>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
    T::Output: std::iter::Sum,
{
    type Output = T::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).sum()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, T> Operation<K> for Product<T>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
    T::Output: std::iter::Product,
{
    type Output = T::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).product()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, T> Operation<K> for MinOf<T>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
{
    type Output = T::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).min().unwrap()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, T> Operation<K> for MaxOf<T>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
{
    type Output = T::Output;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.iter().map(|x| x.call(values)).max().unwrap()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<K, T, F> Operation<K> for Any<T, F>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
    F: Fn(T::Output) -> bool + Clone,
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

impl<K, T, F> Operation<K> for All<T, F>
where
    K: Clone + Copy + Ord,
    T: Operation<K> + Clone,
    F: Fn(T::Output) -> bool + Clone,
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

impl<K, O, F, C, L, R> Operation<K> for Branch<F, C, L, R>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
    F: Fn(C::Output) -> bool + Clone,
    C: Operation<K> + Clone,
    L: Operation<K, Output = O> + Clone,
    R: Operation<K, Output = O> + Clone,
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

impl<K, O> Operation<K> for Boxed<K, O>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
{
    type Output = O;

    fn call(&self, values: &[K]) -> Self::Output {
        self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<K, O> Debug for Boxed<K, O>
where
    K: Clone + Copy + Ord,
    O: Clone + Copy + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Boxed").field(&self.0).finish()
    }
}
