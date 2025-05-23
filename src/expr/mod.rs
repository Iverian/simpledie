pub mod composite;
pub mod ext;
pub mod overloading;

use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::util::DefaultKey;

pub trait Operation: Debug + DynClone {
    fn call(&self, values: &[DefaultKey]) -> DefaultKey;

    fn shift_indices(&mut self, value: usize);

    fn boxed(self) -> Boxed
    where
        Self: Sized + 'static,
    {
        Boxed(Box::new(self))
    }
}

dyn_clone::clone_trait_object!(Operation);

#[derive(Clone, Copy, Debug)]
pub struct Id(usize);

#[derive(Clone, Copy)]
pub struct Map<T, F>(T, F);

#[derive(Clone, Copy, Debug)]
pub struct Neg<T>(T);

#[derive(Clone, Copy, Debug)]
pub struct AddKey<T>(T, DefaultKey);

#[derive(Clone, Copy, Debug)]
pub struct MulKey<T>(T, DefaultKey);

#[derive(Clone, Copy, Debug)]
pub struct DivKey<T>(T, DefaultKey);

#[derive(Clone, Copy, Debug)]
pub struct Not<T>(T);

#[derive(Clone, Debug)]
pub struct Eq<T, const N: usize = 1>(T, [DefaultKey; N]);

#[derive(Clone, Copy, Debug)]
pub struct Cmp<T>(T, DefaultKey);

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

#[derive(Clone, Debug)]
pub struct Boxed(Box<dyn Operation + 'static>);

impl Operation for Id {
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        values[self.0]
    }

    fn shift_indices(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<T, F, O> Operation for Map<T, F>
where
    T: Operation + Clone,
    F: Fn(DefaultKey) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.1(self.0.call(values)).into()
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

impl<T> Operation for Neg<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        -self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for AddKey<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) + self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for MulKey<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) * self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for DivKey<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) / self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for Not<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        DefaultKey::from(match self.0.call(values) {
            0 => 1,
            _ => 0,
        })
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T, const N: usize> Operation for Eq<T, N>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        let v = self.0.call(values);
        DefaultKey::from(self.1.contains(&v))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for Cmp<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values).cmp(&self.1) as DefaultKey
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<L, R> Operation for Add<L, R>
where
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) + self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<L, R> Operation for Mul<L, R>
where
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) * self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<L, R> Operation for Div<L, R>
where
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values) / self.1.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<L, R> Operation for Min<L, R>
where
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values).min(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<L, R> Operation for Max<L, R>
where
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values).max(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<T, F, O> Operation for Fold<T, F>
where
    T: Operation + Clone,
    F: Fn(&[DefaultKey]) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
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

impl<T1, T2, F, O> Operation for FoldTwo<T1, T2, F>
where
    T1: Operation + Clone,
    T2: Operation + Clone,
    F: Fn(DefaultKey, DefaultKey) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.2(self.0.call(values), self.1.call(values)).into()
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

impl<T1, T2, T3, F, O> Operation for FoldThree<T1, T2, T3, F>
where
    T1: Operation + Clone,
    T2: Operation + Clone,
    T3: Operation + Clone,
    F: Fn(DefaultKey, DefaultKey, DefaultKey) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.3(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
        )
        .into()
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

impl<T1, T2, T3, T4, F, O> Operation for FoldFour<T1, T2, T3, T4, F>
where
    T1: Operation + Clone,
    T2: Operation + Clone,
    T3: Operation + Clone,
    T4: Operation + Clone,
    F: Fn(DefaultKey, DefaultKey, DefaultKey, DefaultKey) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
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

impl<T1, T2, T3, T4, T5, F, O> Operation for FoldFive<T1, T2, T3, T4, T5, F>
where
    T1: Operation + Clone,
    T2: Operation + Clone,
    T3: Operation + Clone,
    T4: Operation + Clone,
    T5: Operation + Clone,
    F: Fn(DefaultKey, DefaultKey, DefaultKey, DefaultKey, DefaultKey) -> O + Clone,
    O: Into<DefaultKey>,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.5(
            self.0.call(values),
            self.1.call(values),
            self.2.call(values),
            self.3.call(values),
            self.4.call(values),
        )
        .into()
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

impl<T> Operation for Sum<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.iter().map(|x| x.call(values)).sum()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T> Operation for Product<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.iter().map(|x| x.call(values)).product()
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T> Operation for MinOf<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.iter().map(|x| x.call(values)).min().unwrap_or(0)
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T> Operation for MaxOf<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.iter().map(|x| x.call(values)).max().unwrap_or(0)
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_indices(value);
        }
    }
}

impl<T, F> Operation for Any<T, F>
where
    T: Operation + Clone,
    F: Fn(DefaultKey) -> bool + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0
            .iter()
            .map(|x| x.call(values))
            .any(|x| self.1(x))
            .into()
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

impl<T, F> Operation for All<T, F>
where
    T: Operation + Clone,
    F: Fn(DefaultKey) -> bool + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0
            .iter()
            .map(|x| x.call(values))
            .all(|x| self.1(x))
            .into()
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

impl<F, C, L, R> Operation for Branch<F, C, L, R>
where
    F: Fn(DefaultKey) -> bool + Clone,
    C: Operation + Clone,
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
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

impl Operation for Boxed {
    fn call(&self, values: &[DefaultKey]) -> DefaultKey {
        self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}
