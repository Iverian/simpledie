use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

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

impl<T> Add<Key> for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::AddKey<T>>;

    fn add(self, rhs: Key) -> Self::Output {
        self.kadd(rhs)
    }
}

impl Sub<Key> for Die {
    type Output = Composite<expr::AddKey<Id>>;

    fn sub(self, rhs: Key) -> Self::Output {
        self.ksub(rhs)
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

impl Mul<usize> for Die {
    type Output = Composite<expr::Sum<Id>>;

    fn mul(self, rhs: usize) -> Self::Output {
        self.sum_n(rhs)
    }
}

impl<T> Mul<usize> for Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    type Output = Composite<expr::Sum<T>>;

    fn mul(self, rhs: usize) -> Self::Output {
        self.sum_n(rhs)
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
