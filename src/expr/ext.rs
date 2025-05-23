use std::cmp::{Ord, Ordering};
use std::fmt::Debug;

use super::composite::Composite;
use super::{
    Add, AddKey, All, Any, Boxed, Branch, Cmp, Div, DivKey, Eq, Fold, FoldFive, FoldFour,
    FoldThree, FoldTwo, Id, Map, Max, MaxOf, Min, MinOf, Mul, MulKey, Neg, Not, Operation, Product,
    Sum,
};
use crate::util::{DieList, Key};
use crate::Die;

pub trait Expr: Clone + Debug {
    type Op: Operation + Clone + 'static;

    fn into_composite(self) -> Composite<Self::Op>;

    fn eval(self) -> Die;

    fn map<F, O>(self, op: F) -> Composite<Map<Self::Op, F>>
    where
        F: Fn(Key) -> O + Clone,
        O: Into<Key>,
    {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Map(me.op, op),
        }
    }

    fn neg(self) -> Composite<Neg<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Neg(me.op),
        }
    }

    fn kadd(self, rhs: Key) -> Composite<AddKey<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: AddKey(me.op, rhs),
        }
    }

    fn ksub(self, rhs: Key) -> Composite<AddKey<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: AddKey(me.op, -rhs),
        }
    }

    fn kmul(self, rhs: Key) -> Composite<MulKey<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: MulKey(me.op, rhs),
        }
    }

    fn kdiv(self, rhs: Key) -> Composite<DivKey<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: DivKey(me.op, rhs),
        }
    }

    fn not(self) -> Composite<Not<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Not(me.op),
        }
    }

    fn contains<const N: usize>(self, rhs: [Key; N]) -> Composite<Eq<Self::Op, N>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Eq(me.op, rhs),
        }
    }

    fn eq(self, rhs: Key) -> Composite<Eq<Self::Op>> {
        self.contains([rhs])
    }

    fn neq(self, rhs: Key) -> Composite<Not<Eq<Self::Op>>> {
        self.eq(rhs).not()
    }

    fn cmp(self, rhs: Key) -> Composite<Cmp<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Cmp(me.op, rhs),
        }
    }

    fn lt(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs).eq(Ordering::Less as Key)
    }

    fn le(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>, 2>> {
        self.cmp(rhs)
            .contains([Ordering::Less as Key, Ordering::Equal as Key])
    }

    fn gt(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs).eq(Ordering::Greater as Key)
    }

    fn ge(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>, 2>> {
        self.cmp(rhs)
            .contains([Ordering::Equal as Key, Ordering::Greater as Key])
    }

    fn add<T>(self, rhs: T) -> Composite<Add<Self::Op, T::Op>>
    where
        T: Expr,
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

    fn sub<T>(self, rhs: T) -> Composite<Add<Self::Op, Neg<T::Op>>>
    where
        T: Expr,
    {
        self.add(rhs.into_composite().neg())
    }

    fn mul<T>(self, rhs: T) -> Composite<Mul<Self::Op, T::Op>>
    where
        T: Expr,
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

    fn div<T>(self, rhs: T) -> Composite<Div<Self::Op, T::Op>>
    where
        T: Expr,
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

    fn min<T>(self, rhs: T) -> Composite<Min<Self::Op, T::Op>>
    where
        T: Expr,
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

    fn max<T>(self, rhs: T) -> Composite<Max<Self::Op, T::Op>>
    where
        T: Expr,
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

    fn fold_n<F, O>(self, size: usize, op: F) -> Die
    where
        F: Fn(Key, Key) -> O,
        O: Into<Key>,
    {
        self.eval().eval_n(size, |x, y| op(x, y).into())
    }

    fn sum_n(self, size: usize) -> Die {
        self.eval().eval_n(size, |x, y| x + y)
    }

    fn product_n(self, size: usize) -> Die {
        self.eval().eval_n(size, |x, y| x * y)
    }

    fn min_of_n(self, size: usize) -> Die {
        self.eval().eval_n(size, Ord::min)
    }

    fn max_of_n(self, size: usize) -> Die {
        self.eval().eval_n(size, Ord::max)
    }

    fn any_n<F>(self, size: usize, pred: F) -> Die
    where
        F: Fn(Key) -> bool,
    {
        self.eval()
            .eval_n(size, |x, y| Key::from(pred(x) || pred(y)))
    }

    fn all_n<F>(self, size: usize, pred: F) -> Die
    where
        F: Fn(Key) -> bool,
    {
        self.eval()
            .eval_n(size, |x, y| Key::from(pred(x) && pred(y)))
    }

    #[allow(clippy::type_complexity)]
    fn branch<F, L, R>(
        self,
        pred: F,
        lhs: L,
        rhs: R,
    ) -> Composite<Branch<F, Self::Op, L::Op, R::Op>>
    where
        F: Fn(Key) -> bool + Clone,
        L: Expr,
        R: Expr,
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

    fn boxed(self) -> Composite<Boxed>
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
pub struct DynFoldBuilder(DieList, Vec<Boxed>);

impl Die {
    #[must_use]
    pub fn dyn_fold() -> DynFoldBuilder {
        DynFoldBuilder::default()
    }

    pub fn fold_two<T1, T2, F, O>(e1: T1, e2: T2, op: F) -> Composite<FoldTwo<T1::Op, T2::Op, F>>
    where
        T1: Expr,
        T2: Expr,
        F: Fn(Key, Key) -> O + Clone,
        O: Into<Key>,
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
    pub fn fold_three<T1, T2, T3, F, O>(
        e1: T1,
        e2: T2,
        e3: T3,
        op: F,
    ) -> Composite<FoldThree<T1::Op, T2::Op, T3::Op, F>>
    where
        T1: Expr,
        T2: Expr,
        T3: Expr,
        F: Fn(Key, Key, Key) -> O + Clone,
        O: Into<Key>,
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
    pub fn fold_four<T1, T2, T3, T4, F, O>(
        e1: T1,
        e2: T2,
        e3: T3,
        e4: T4,
        op: F,
    ) -> Composite<FoldFour<T1::Op, T2::Op, T3::Op, T4::Op, F>>
    where
        T1: Expr,
        T2: Expr,
        T3: Expr,
        T4: Expr,
        F: Fn(Key, Key, Key, Key) -> O + Clone,
        O: Into<Key>,
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
    pub fn fold_five<T1, T2, T3, T4, T5, F, O>(
        e1: T1,
        e2: T2,
        e3: T3,
        e4: T4,
        e5: T5,
        op: F,
    ) -> Composite<FoldFive<T1::Op, T2::Op, T3::Op, T4::Op, T5::Op, F>>
    where
        T1: Expr,
        T2: Expr,
        T3: Expr,
        T4: Expr,
        T5: Expr,
        F: Fn(Key, Key, Key, Key, Key) -> O + Clone,
        O: Into<Key>,
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
    pub fn fold<F, I, E>(iter: I, op: F) -> Composite<Fold<E::Op, F>>
    where
        F: Fn(&[Key]) -> Key + Clone,
        I: IntoIterator<Item = E>,
        E: Expr,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Fold(items, op),
        }
    }

    pub fn sum<I, E>(iter: I) -> Composite<Sum<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Sum(items),
        }
    }

    pub fn product<I, E>(iter: I) -> Composite<Product<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Product(items),
        }
    }

    pub fn min_of<I, E>(iter: I) -> Composite<MinOf<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: MinOf(items),
        }
    }

    pub fn max_of<I, E>(iter: I) -> Composite<MaxOf<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: MaxOf(items),
        }
    }

    pub fn any<I, E, F>(iter: I, pred: F) -> Composite<Any<E::Op, F>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
        F: Fn(Key) -> bool + Clone,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: Any(items, pred),
        }
    }

    pub fn all<I, E, F>(iter: I, pred: F) -> Composite<All<E::Op, F>>
    where
        I: IntoIterator<Item = E>,
        E: Expr,
        F: Fn(Key) -> bool + Clone,
    {
        let (dice, items) = Self::parts(iter);
        Composite {
            dice,
            op: All(items, pred),
        }
    }

    fn parts<I, E>(iter: I) -> (Vec<Die>, Vec<E::Op>)
    where
        I: IntoIterator<Item = E>,
        E: Expr,
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

impl<T> Expr for Composite<T>
where
    T: Operation + Clone + 'static,
{
    type Op = T;

    fn into_composite(self) -> Composite<Self::Op> {
        self
    }

    fn eval(self) -> Die {
        self.eval()
    }
}

impl Expr for Die {
    type Op = Id;

    fn into_composite(self) -> Composite<Self::Op> {
        Composite {
            dice: vec![self],
            op: Id(0),
        }
    }

    fn eval(self) -> Die {
        self
    }
}

impl DynFoldBuilder {
    #[must_use]
    pub fn push<T>(mut self, expr: T) -> Self
    where
        T: Expr,
    {
        let mut expr = expr.into_composite();
        self.0.extend(expr.dice);
        expr.op.shift_indices(self.0.len());
        self.1.push(expr.op.boxed());
        self
    }

    #[must_use]
    pub fn extend<I, T>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Expr,
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
    pub fn repeat<T>(mut self, e: T, size: usize) -> Self
    where
        T: Expr,
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
    pub fn build<F>(self, op: F) -> Composite<Fold<Boxed, F>>
    where
        F: Fn(&[Key]) -> Key + Clone,
    {
        Composite {
            dice: self.0,
            op: Fold(self.1, op),
        }
    }
}
