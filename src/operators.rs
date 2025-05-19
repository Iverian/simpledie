use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::expr::{Composite, Expr, Id, Operation};
use crate::util::Key;
use crate::{expr, Die};

impl Neg for Die {
    type Output = Composite<expr::Neg<Id>>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

impl<T> Neg for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::Neg<T>>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

impl Add<Key> for Die {
    type Output = Composite<expr::AddKey<Id>>;

    fn add(self, rhs: Key) -> Self::Output {
        self.kadd(rhs)
    }
}

impl Add<Die> for Key {
    type Output = Composite<expr::AddKey<Id>>;

    fn add(self, rhs: Die) -> Self::Output {
        rhs.kadd(self)
    }
}

impl<T> Add<Key> for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::AddKey<T>>;

    fn add(self, rhs: Key) -> Self::Output {
        self.kadd(rhs)
    }
}

impl<T> Add<Composite<T>> for Key
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::AddKey<T>>;

    fn add(self, rhs: Composite<T>) -> Self::Output {
        rhs.kadd(self)
    }
}

impl Sub<Key> for Die {
    type Output = Composite<expr::AddKey<Id>>;

    fn sub(self, rhs: Key) -> Self::Output {
        self.ksub(rhs)
    }
}

impl Sub<Die> for Key {
    type Output = Composite<expr::AddKey<expr::Neg<Id>>>;

    fn sub(self, rhs: Die) -> Self::Output {
        Expr::neg(rhs).kadd(self)
    }
}

impl<T> Sub<Key> for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::AddKey<T>>;

    fn sub(self, rhs: Key) -> Self::Output {
        self.ksub(rhs)
    }
}

impl<T> Sub<Composite<T>> for Key
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::AddKey<expr::Neg<T>>>;

    fn sub(self, rhs: Composite<T>) -> Self::Output {
        Expr::neg(rhs).kadd(self)
    }
}

impl Mul<Die> for usize {
    type Output = Composite<expr::Sum<Id>>;

    fn mul(self, rhs: Die) -> Self::Output {
        rhs.sum_n(self)
    }
}

impl<T> Mul<Composite<T>> for usize
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::Sum<T>>;

    fn mul(self, rhs: Composite<T>) -> Self::Output {
        rhs.sum_n(self)
    }
}

impl Div<Die> for usize {
    type Output = Composite<expr::MaxOf<Id>>;

    fn div(self, rhs: Die) -> Self::Output {
        rhs.max_of_n(self)
    }
}

impl<T> Div<Composite<T>> for usize
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::MaxOf<T>>;

    fn div(self, rhs: Composite<T>) -> Self::Output {
        rhs.max_of_n(self)
    }
}

impl<R> Add<R> for Die
where
    R: Expr,
{
    type Output = Composite<expr::Add<Id, R::Op>>;

    fn add(self, rhs: R) -> Self::Output {
        Expr::add(self, rhs)
    }
}

impl<T, R> Add<R> for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
    R: Expr,
{
    type Output = Composite<expr::Add<T, R::Op>>;

    fn add(self, rhs: R) -> Self::Output {
        Expr::add(self, rhs)
    }
}
