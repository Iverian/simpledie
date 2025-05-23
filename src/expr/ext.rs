use std::cmp::{Ord, Ordering};
use std::fmt::Debug;

use super::composite::Composite;
use super::{
    Add, AddKey, All, Any, Boxed, Branch, Cmp, Div, DivKey, Eq, Fold, FoldFive, FoldFour,
    FoldThree, FoldTwo, Id, Map, Max, MaxOf, Min, MinOf, Mul, MulKey, Neg, Not, Operation, Product,
    RawOperation, Sub, SubKey, Sum,
};
use crate::util::{DefaultKey, DieList};
use crate::{Die, Key};

pub trait Expr<K = DefaultKey>: Clone + Debug
where
    K: Key,
{
    type Op: Operation<K>;

    fn into_composite(self) -> Composite<Self::Op, K>;

    fn eval(self) -> Die<<Self::Op as RawOperation<K>>::Output>;

    fn map<F, V>(self, op: F) -> Composite<Map<Self::Op, F>, K>
    where
        V: Key,
        F: Fn(<Self::Op as RawOperation<K>>::Output) -> V + Clone,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Map(me.op, op),
        }
    }

    fn neg(self) -> Composite<Neg<Self::Op>, K>
    where
        <Self::Op as RawOperation<K>>::Output: std::ops::Neg,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Neg>::Output: Key,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Neg(me.op),
        }
    }

    fn kadd<R>(self, rhs: R) -> Composite<AddKey<Self::Op, R>, K>
    where
        <Self::Op as RawOperation<K>>::Output: std::ops::Add<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Add<R>>::Output: Key,
        R: Key,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: AddKey(me.op, rhs),
        }
    }

    fn ksub<R>(self, rhs: R) -> Composite<SubKey<Self::Op, R>, K>
    where
        R: Key,
        <Self::Op as RawOperation<K>>::Output: std::ops::Sub<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Sub<R>>::Output: Key,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: SubKey(me.op, rhs),
        }
    }

    fn kmul<R>(self, rhs: R) -> Composite<MulKey<Self::Op, R>, K>
    where
        R: Key,
        <Self::Op as RawOperation<K>>::Output: std::ops::Mul<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Mul<R>>::Output: Key,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: MulKey(me.op, rhs),
        }
    }

    fn kdiv<R>(self, rhs: R) -> Composite<DivKey<Self::Op, R>, K>
    where
        R: Key,
        <Self::Op as RawOperation<K>>::Output: std::ops::Div<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Div<R>>::Output: Key,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: DivKey(me.op, rhs),
        }
    }

    fn not(self) -> Composite<Not<Self::Op>, K>
    where
        <Self::Op as RawOperation<K>>::Output: Into<bool>,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Not(me.op),
        }
    }

    fn contains<R, const N: usize>(self, rhs: [R; N]) -> Composite<Eq<Self::Op, R, N>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Eq(me.op, rhs),
        }
    }

    fn eq<R>(self, rhs: R) -> Composite<Eq<Self::Op, R>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.contains([rhs])
    }

    fn neq<R>(self, rhs: R) -> Composite<Not<Eq<Self::Op, R>>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.eq(rhs).not()
    }

    fn cmp<R>(self, rhs: R) -> Composite<Cmp<Self::Op, R>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Cmp(me.op, rhs),
        }
    }

    fn lt<R>(self, rhs: R) -> Composite<Eq<Cmp<Self::Op, R>, Ordering>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.cmp(rhs).eq(Ordering::Less)
    }

    fn le<R>(self, rhs: R) -> Composite<Eq<Cmp<Self::Op, R>, Ordering, 2>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.cmp(rhs).contains([Ordering::Less, Ordering::Equal])
    }

    fn gt<R>(self, rhs: R) -> Composite<Eq<Cmp<Self::Op, R>, Ordering>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.cmp(rhs).eq(Ordering::Greater)
    }

    fn ge<R>(self, rhs: R) -> Composite<Eq<Cmp<Self::Op, R>, Ordering, 2>, K>
    where
        R: Copy + Debug + Into<<Self::Op as RawOperation<K>>::Output>,
    {
        self.cmp(rhs).contains([Ordering::Equal, Ordering::Greater])
    }

    fn add<R, O, T>(self, rhs: T) -> Composite<Add<Self::Op, O>, K>
    where
        R: Key,
        O: Operation<K, Output = R>,
        T: Expr<K, Op = O>,
        <Self::Op as RawOperation<K>>::Output: std::ops::Add<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Add<R>>::Output: Key,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Add(me.op, rhs.op),
        }
    }

    fn sub<R, O, T>(self, rhs: T) -> Composite<Sub<Self::Op, O>, K>
    where
        R: Key,
        O: Operation<K, Output = R>,
        T: Expr<K, Op = O>,
        <Self::Op as RawOperation<K>>::Output: std::ops::Sub<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Sub<R>>::Output: Key,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Sub(me.op, rhs.op),
        }
    }

    fn mul<R, O, T>(self, rhs: T) -> Composite<Mul<Self::Op, O>, K>
    where
        R: Key,
        O: Operation<K, Output = R>,
        T: Expr<K, Op = O>,
        <Self::Op as RawOperation<K>>::Output: std::ops::Mul<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Mul<R>>::Output: Key,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Mul(me.op, rhs.op),
        }
    }

    fn div<R, O, T>(self, rhs: T) -> Composite<Div<Self::Op, O>, K>
    where
        R: Key,
        O: Operation<K, Output = R>,
        T: Expr<K, Op = O>,
        <Self::Op as RawOperation<K>>::Output: std::ops::Div<R>,
        <<Self::Op as RawOperation<K>>::Output as std::ops::Div<R>>::Output: Key,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Div(me.op, rhs.op),
        }
    }

    fn min<O, T>(self, rhs: T) -> Composite<Min<Self::Op, O>, K>
    where
        O: Operation<K, Output = <Self::Op as RawOperation<K>>::Output>,
        T: Expr<K, Op = O>,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Min(me.op, rhs.op),
        }
    }

    fn max<O, T>(self, rhs: T) -> Composite<Max<Self::Op, O>, K>
    where
        O: Operation<K, Output = <Self::Op as RawOperation<K>>::Output>,
        T: Expr<K, Op = O>,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: Max(me.op, rhs.op),
        }
    }

    fn fold_n<F>(self, size: usize, op: F) -> Die<<Self::Op as RawOperation<K>>::Output>
    where
        F: Fn(
            <Self::Op as RawOperation<K>>::Output,
            <Self::Op as RawOperation<K>>::Output,
        ) -> <Self::Op as RawOperation<K>>::Output,
    {
        self.eval().eval_n(size, op)
    }

    fn sum_n(
        self,
        size: usize,
    ) -> Die<<<Self::Op as RawOperation<K>>::Output as std::ops::Add>::Output>
    where
        <Self::Op as RawOperation<K>>::Output:
            std::ops::Add<Output = <Self::Op as RawOperation<K>>::Output>,
    {
        self.eval().eval_n(size, |x, y| x + y)
    }

    fn product_n(self, size: usize) -> Die<<Self::Op as RawOperation<K>>::Output>
    where
        <Self::Op as RawOperation<K>>::Output:
            std::ops::Mul<Output = <Self::Op as RawOperation<K>>::Output>,
    {
        self.eval().eval_n(size, |x, y| x * y)
    }

    fn min_of_n(self, size: usize) -> Die<<Self::Op as RawOperation<K>>::Output> {
        self.eval().eval_n(size, Ord::min)
    }

    fn max_of_n(self, size: usize) -> Die<<Self::Op as RawOperation<K>>::Output> {
        self.eval().eval_n(size, Ord::max)
    }

    fn any_n<F>(self, size: usize, pred: F) -> Composite<Any<Self::Op, F>, K>
    where
        F: Fn(<Self::Op as RawOperation<K>>::Output) -> bool + Clone,
    {
        let (dice, items) = self.into_composite().explode(size);
        Composite {
            dice,
            op: Any(items, pred),
        }
    }

    fn all_n<F>(self, size: usize, pred: F) -> Composite<All<Self::Op, F>, K>
    where
        F: Fn(<Self::Op as RawOperation<K>>::Output) -> bool + Clone,
    {
        let (dice, items) = self.into_composite().explode(size);
        Composite {
            dice,
            op: All(items, pred),
        }
    }

    #[allow(clippy::type_complexity)]
    fn branch<V, LO, RO, L, R, F>(
        self,
        pred: F,
        lhs: L,
        rhs: R,
    ) -> Composite<Branch<F, Self::Op, LO, RO>, K>
    where
        V: Key,
        LO: Operation<K, Output = V>,
        RO: Operation<K, Output = V>,
        L: Expr<K, Op = LO>,
        R: Expr<K, Op = RO>,
        F: Fn(<Self::Op as RawOperation<K>>::Output) -> bool + Clone,
    {
        let mut me = self.into_composite();
        let mut lhs = lhs.into_composite();
        let mut rhs = rhs.into_composite();

        lhs.op.shift_indices(me.dice.len());
        me.dice.extend(lhs.dice);
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);

        Composite {
            dice: me.dice,
            op: Branch(pred, me.op, lhs.op, rhs.op),
        }
    }

    fn boxed(self) -> Composite<Boxed<K, <Self::Op as RawOperation<K>>::Output>, K>
    where
        Self::Op: 'static,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: me.op.boxed(),
        }
    }
}

#[derive(Clone, Default)]
pub struct DynFoldBuilder<T = DefaultKey, O = DefaultKey>(DieList<T>, Vec<Boxed<T, O>>)
where
    T: Key,
    O: Key;

impl Die {
    #[must_use]
    pub fn dyn_fold() -> DynFoldBuilder {
        DynFoldBuilder::default()
    }

    #[allow(clippy::type_complexity)]
    pub fn fold_two<K, R1, R2, V, O1, O2, T1, T2, F>(
        e1: T1,
        e2: T2,
        op: F,
    ) -> Composite<FoldTwo<O1, O2, F>, K>
    where
        K: Key,
        R1: Key,
        R2: Key,
        V: Key,
        O1: Operation<K, Output = R1>,
        O2: Operation<K, Output = R2>,
        T1: Expr<K, Op = O1>,
        T2: Expr<K, Op = O2>,
        F: Fn(R1, R2) -> V + Clone,
    {
        let e1 = e1.into_composite();
        let mut e2 = e2.into_composite();

        let mut dice = e1.dice;
        e2.op.shift_indices(dice.len());
        dice.extend(e2.dice);

        Composite {
            dice,
            op: FoldTwo(e1.op, e2.op, op),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn fold_three<K, R1, R2, R3, V, O1, O2, O3, T1, T2, T3, F>(
        e1: T1,
        e2: T2,
        e3: T3,
        op: F,
    ) -> Composite<FoldThree<O1, O2, O3, F>, K>
    where
        K: Key,
        R1: Key,
        R2: Key,
        R3: Key,
        V: Key,
        O1: Operation<K, Output = R1>,
        O2: Operation<K, Output = R2>,
        O3: Operation<K, Output = R3>,
        T1: Expr<K, Op = O1>,
        T2: Expr<K, Op = O2>,
        T3: Expr<K, Op = O3>,
        F: Fn(R1, R2, R3) -> V + Clone,
    {
        let e1 = e1.into_composite();
        let mut e2 = e2.into_composite();
        let mut e3 = e3.into_composite();

        let mut dice = e1.dice;
        dice.reserve(e2.dice.len() + e3.dice.len());
        e2.op.shift_indices(dice.len());
        dice.extend(e2.dice);
        e3.op.shift_indices(dice.len());
        dice.extend(e3.dice);

        Composite {
            dice,
            op: FoldThree(e1.op, e2.op, e3.op, op),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn fold_four<K, R1, R2, R3, R4, V, O1, O2, O3, O4, T1, T2, T3, T4, F>(
        e1: T1,
        e2: T2,
        e3: T3,
        e4: T4,
        op: F,
    ) -> Composite<FoldFour<O1, O2, O3, O4, F>, K>
    where
        K: Key,
        R1: Key,
        R2: Key,
        R3: Key,
        R4: Key,
        V: Key,
        O1: Operation<K, Output = R1>,
        O2: Operation<K, Output = R2>,
        O3: Operation<K, Output = R3>,
        O4: Operation<K, Output = R4>,
        T1: Expr<K, Op = O1>,
        T2: Expr<K, Op = O2>,
        T3: Expr<K, Op = O3>,
        T4: Expr<K, Op = O4>,
        F: Fn(R1, R2, R3, R4) -> V + Clone,
    {
        let e1 = e1.into_composite();
        let mut e2 = e2.into_composite();
        let mut e3 = e3.into_composite();
        let mut e4 = e4.into_composite();

        let mut dice = e1.dice;
        dice.reserve(e2.dice.len() + e3.dice.len() + e4.dice.len());
        e2.op.shift_indices(dice.len());
        dice.extend(e2.dice);
        e3.op.shift_indices(dice.len());
        dice.extend(e3.dice);
        e4.op.shift_indices(dice.len());
        dice.extend(e4.dice);

        Composite {
            dice,
            op: FoldFour(e1.op, e2.op, e3.op, e4.op, op),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn fold_five<K, R1, R2, R3, R4, R5, V, O1, O2, O3, O4, O5, T1, T2, T3, T4, T5, F>(
        e1: T1,
        e2: T2,
        e3: T3,
        e4: T4,
        e5: T5,
        op: F,
    ) -> Composite<FoldFive<O1, O2, O3, O4, O5, F>, K>
    where
        K: Key,
        R1: Key,
        R2: Key,
        R3: Key,
        R4: Key,
        R5: Key,
        V: Key,
        O1: Operation<K, Output = R1>,
        O2: Operation<K, Output = R2>,
        O3: Operation<K, Output = R3>,
        O4: Operation<K, Output = R4>,
        O5: Operation<K, Output = R5>,
        T1: Expr<K, Op = O1>,
        T2: Expr<K, Op = O2>,
        T3: Expr<K, Op = O3>,
        T4: Expr<K, Op = O4>,
        T5: Expr<K, Op = O5>,
        F: Fn(R1, R2, R3, R4, R5) -> V + Clone,
    {
        let e1 = e1.into_composite();
        let mut e2 = e2.into_composite();
        let mut e3 = e3.into_composite();
        let mut e4 = e4.into_composite();
        let mut e5 = e5.into_composite();

        let mut dice = e1.dice;
        dice.reserve(e2.dice.len() + e3.dice.len() + e4.dice.len() + e5.dice.len());
        e2.op.shift_indices(dice.len());
        dice.extend(e2.dice);
        e3.op.shift_indices(dice.len());
        dice.extend(e3.dice);
        e4.op.shift_indices(dice.len());
        dice.extend(e4.dice);
        e5.op.shift_indices(dice.len());
        dice.extend(e5.dice);

        Composite {
            dice,
            op: FoldFive(e1.op, e2.op, e3.op, e4.op, e5.op, op),
        }
    }

    #[must_use]
    pub fn fold<K, R, O, V, I, E, F>(iter: I, op: F) -> Composite<Fold<E::Op, F>, K>
    where
        K: Key,
        R: Key,
        V: Key,
        O: Operation<K, Output = R>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
        F: Fn(&[R]) -> V + Clone,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Fold(items, op),
        }
    }

    pub fn sum<K, R, O, I, E>(iter: I) -> Composite<Sum<E::Op>, K>
    where
        K: Key,
        R: Key + std::iter::Sum,
        O: Operation<K, Output = R>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Sum(items),
        }
    }

    pub fn product<K, R, O, I, E>(iter: I) -> Composite<Product<E::Op>, K>
    where
        K: Key,
        R: Key + std::iter::Product,
        O: Operation<K, Output = R>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Product(items),
        }
    }

    pub fn min_of<K, O, I, E>(iter: I) -> Composite<MinOf<E::Op>, K>
    where
        K: Key,
        O: Operation<K>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: MinOf(items),
        }
    }

    pub fn max_of<K, O, I, E>(iter: I) -> Composite<MaxOf<E::Op>, K>
    where
        K: Key,
        O: Operation<K>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: MaxOf(items),
        }
    }

    pub fn any<K, O, I, E, F>(iter: I, pred: F) -> Composite<Any<E::Op, F>, K>
    where
        K: Key,
        O: Operation<K>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
        F: Fn(O::Output) -> bool + Clone,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Any(items, pred),
        }
    }

    pub fn all<K, O, I, E, F>(iter: I, pred: F) -> Composite<All<E::Op, F>, K>
    where
        K: Key,
        O: Operation<K>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
        F: Fn(O::Output) -> bool + Clone,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: All(items, pred),
        }
    }

    fn parts<K, O, I, E>(iter: I) -> (Vec<Die<K>>, Vec<O>)
    where
        K: Key,
        O: Operation<K>,
        I: IntoIterator<Item = E>,
        E: Expr<K, Op = O>,
    {
        let iter = iter.into_iter();
        let (l, u) = iter.size_hint();
        let s = u.unwrap_or(l);
        let mut dice = Vec::with_capacity(s);
        let mut items = Vec::with_capacity(s);
        for i in iter {
            let mut i = i.into_composite();
            i.op.shift_indices(dice.len());
            dice.extend(i.dice);
            items.push(i.op);
        }
        (dice, items)
    }
}

impl<K, T> Expr<K> for Composite<T, K>
where
    K: Key,
    T: Operation<K>,
{
    type Op = T;

    fn into_composite(self) -> Composite<Self::Op, K> {
        self
    }

    fn eval(self) -> Die<T::Output> {
        self.eval()
    }
}

impl<K> Expr<K> for Die<K>
where
    K: Key,
{
    type Op = Id;

    fn into_composite(self) -> Composite<Self::Op, K> {
        Composite {
            dice: vec![self],
            op: Id(0),
        }
    }

    fn eval(self) -> Die<K> {
        self
    }
}

impl<K, R> DynFoldBuilder<K, R>
where
    K: Key,
    R: Key,
{
    #[must_use]
    pub fn push<O, T>(mut self, expr: T) -> Self
    where
        O: Operation<K, Output = R> + 'static,
        T: Expr<K, Op = O>,
    {
        let mut expr = expr.into_composite();
        self.0.extend(expr.dice);
        expr.op.shift_indices(self.0.len());
        self.1.push(expr.op.boxed());
        self
    }

    #[must_use]
    pub fn extend<I, O, T>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        O: Operation<K, Output = R> + 'static,
        T: Expr<K, Op = O>,
    {
        for i in iter {
            let mut i = i.into_composite();
            i.op.shift_indices(self.0.len());
            self.0.extend(i.dice);
            self.1.push(i.op.boxed());
        }
        self
    }

    #[must_use]
    pub fn repeat<O, T>(mut self, e: T, size: usize) -> Self
    where
        O: Operation<K, Output = R> + 'static,
        T: Expr<K, Op = O>,
    {
        if size == 0 {
            return self;
        }
        let mut e = e.into_composite();

        for _ in 1..size {
            let mut i = e.clone();
            i.op.shift_indices(self.0.len());
            self.0.extend(i.dice);
            self.1.push(i.op.boxed());
        }

        e.op.shift_indices(self.0.len());
        self.0.extend(e.dice);
        self.1.push(e.op.boxed());

        self
    }

    #[must_use]
    pub fn build<V, F>(self, op: F) -> Composite<Fold<Boxed<K, R>, F>, K>
    where
        V: Key,
        F: Fn(&[R]) -> V + Clone,
    {
        Composite {
            dice: self.0,
            op: Fold(self.1, op),
        }
    }
}
