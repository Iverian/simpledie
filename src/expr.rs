use std::cmp::Ordering;

use dyn_clone::DynClone;

use crate::approx::Approx;
use crate::util::{BigUint, DieList, Key, OverflowResult, Value};
use crate::Die;

type OpPtr = Box<dyn Operation + 'static>;

#[derive(Clone, Debug)]
pub struct Composite<T = Index>
where
    T: Operation + Clone + 'static,
{
    dice: DieList,
    op: T,
}

#[derive(Clone, Copy, Debug)]
pub struct Index(usize);

#[derive(Clone, Copy, Debug)]
pub struct Map<T, F>(T, F);

#[derive(Clone, Copy, Debug)]
pub struct Negate<T>(T);

#[derive(Clone, Copy, Debug)]
pub struct AddKey<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct MulKey<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct DivKey<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct Not<T>(T);

#[derive(Clone, Debug)]
pub struct Eq<T>(T, Vec<Key>);

#[derive(Clone, Copy, Debug)]
pub struct Cmp<T>(T, Key);

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

#[derive(Clone, Debug)]
pub struct Fold<T, F>(Vec<T>, F);

#[derive(Clone, Copy, Debug)]
pub struct FoldTwo<L, R, F>(L, R, F);

#[derive(Clone, Copy, Debug)]
pub struct FoldThree<T1, T2, T3, F>(T1, T2, T3, F);

#[derive(Clone)]
pub struct DynFold<F>(Vec<OpPtr>, F);

#[derive(Default)]
pub struct DynFoldBuilder(DieList, Vec<OpPtr>);

#[derive(Clone, Debug)]
pub struct Sum<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct Product<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct MaxOf<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct MinOf<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct Any<T, F>(Vec<T>, F);

#[derive(Clone, Debug)]
pub struct All<T, F>(Vec<T>, F);

#[derive(Clone, Debug, Copy)]
pub struct Branch<F, C, L, R>(F, C, L, R);

#[derive(Clone)]
pub struct Boxed(OpPtr);

pub trait Operation: DynClone {
    fn call(&self, values: &[Key]) -> Key;

    fn shift_indices(&mut self, value: usize);

    fn into_ptr(self) -> OpPtr
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

dyn_clone::clone_trait_object!(Operation);

pub trait Expr: Sized {
    type Op: Operation + Clone + 'static;

    fn into_composite(self) -> Composite<Self::Op>;

    fn eval(self) -> Die {
        self.into_composite().eval()
    }

    fn try_eval(self) -> OverflowResult<Die> {
        self.into_composite().try_eval()
    }

    fn approx_eval(self, approx: Approx) -> Die {
        self.into_composite().approx_eval(approx)
    }

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

    fn neg(self) -> Composite<Negate<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Negate(me.op),
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

    fn contains(self, rhs: Vec<Key>) -> Composite<Eq<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Eq(me.op, rhs),
        }
    }

    fn eq(self, rhs: Key) -> Composite<Eq<Self::Op>> {
        self.contains(vec![rhs])
    }

    fn neq(self, rhs: Key) -> Composite<Not<Eq<Self::Op>>>
    where
        Self::Op: 'static,
    {
        self.eq(rhs).not()
    }

    fn cmp(self, rhs: Key) -> Composite<Cmp<Self::Op>> {
        let me = self.into_composite();
        Composite {
            dice: me.dice,
            op: Cmp(me.op, rhs),
        }
    }

    fn lt(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>>
    where
        Self::Op: 'static,
    {
        self.cmp(rhs).eq(Ordering::Less as Key)
    }

    fn le(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs)
            .contains(vec![Ordering::Less as Key, Ordering::Equal as Key])
    }

    fn gt(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs).eq(Ordering::Greater as Key)
    }

    fn ge(self, rhs: Key) -> Composite<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs)
            .contains(vec![Ordering::Greater as Key, Ordering::Equal as Key])
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

    fn sub<T>(self, rhs: T) -> Composite<Add<Self::Op, Negate<T::Op>>>
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

    fn fold_n<F, O>(self, size: usize, op: F) -> Composite<Fold<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(&[Key]) -> O + Clone,
        O: Into<Key>,
    {
        let (dice, expr) = self.into_composite().explode(size);
        Composite {
            dice,
            op: Fold(expr, op),
        }
    }

    fn fold_two<T, F, O>(self, rhs: T, op: F) -> Composite<FoldTwo<Self::Op, T::Op, F>>
    where
        T: Expr,
        F: Fn(Key, Key) -> O + Clone,
        O: Into<Key>,
    {
        let mut me = self.into_composite();
        let mut rhs = rhs.into_composite();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Composite {
            dice: me.dice,
            op: FoldTwo(me.op, rhs.op, op),
        }
    }

    #[allow(clippy::type_complexity)]
    fn fold_three<T1, T2, F, O>(
        self,
        f: T1,
        s: T2,
        op: F,
    ) -> Composite<FoldThree<Self::Op, T1::Op, T2::Op, F>>
    where
        T1: Expr,
        T2: Expr,
        F: Fn(Key, Key, Key) -> O + Clone,
        O: Into<Key>,
    {
        let mut me = self.into_composite();
        let mut f = f.into_composite();
        let mut s = s.into_composite();
        f.op.shift_indices(me.dice.len());
        me.dice.extend(f.dice);
        s.op.shift_indices(me.dice.len());
        me.dice.extend(s.dice);
        Composite {
            dice: me.dice,
            op: FoldThree(me.op, f.op, s.op, op),
        }
    }

    fn dyn_fold(self) -> DynFoldBuilder {
        let me = self.into_composite();
        DynFoldBuilder(me.dice, vec![me.op.into_ptr()])
    }

    fn sum_n(self, size: usize) -> Composite<Sum<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite { dice, op: Sum(op) }
    }

    fn product_n(self, size: usize) -> Composite<Product<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite {
            dice,
            op: Product(op),
        }
    }

    fn min_of_n(self, size: usize) -> Composite<MinOf<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite {
            dice,
            op: MinOf(op),
        }
    }

    fn max_of_n(self, size: usize) -> Composite<MaxOf<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite {
            dice,
            op: MaxOf(op),
        }
    }

    fn any_n<F>(self, size: usize, pred: F) -> Composite<Any<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(Key) -> bool + Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite {
            dice,
            op: Any(op, pred),
        }
    }

    fn all_n<F>(self, size: usize, pred: F) -> Composite<All<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(Key) -> bool + Clone,
    {
        let (dice, op) = self.into_composite().explode(size);
        Composite {
            dice,
            op: All(op, pred),
        }
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
            op: Boxed(me.op.into_ptr()),
        }
    }
}

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

impl<T> Composite<T>
where
    T: Operation + Clone + 'static,
{
    pub fn eval(self) -> Die {
        Die::eval(self.dice, move |x| self.op.call(x))
    }

    pub fn try_eval(self) -> OverflowResult<Die> {
        Die::try_eval(self.dice, move |x| self.op.call(x))
    }

    pub fn approx_eval(self, approx: Approx) -> Die {
        Die::approx_eval(approx, &self.dice, move |x| self.op.call(x))
    }

    pub fn denom(&self) -> BigUint {
        self.dice
            .iter()
            .map(|x| BigUint::from(x.denom()))
            .fold(BigUint::from(1 as Value), |acc, x| acc * x)
    }

    pub fn can_eval_directly(&self) -> bool {
        self.denom() < BigUint::from(Value::MAX)
    }
}

impl<T> Composite<T>
where
    T: Operation + Clone,
{
    fn explode(self, size: usize) -> (DieList, Vec<T>) {
        assert!(size != 0, "explode size cannot be zero");
        if size == 1 {
            return (self.dice, vec![self.op]);
        }

        let m = self.dice.len();
        let mut dice = Vec::with_capacity(size * m);
        let mut op = Vec::with_capacity(size);

        for i in 1..size {
            let mut item = self.op.clone();
            dice.extend(self.dice.iter().cloned());
            item.shift_indices(i * m);
            op.push(item);
        }
        dice.extend(self.dice);
        op.push(self.op);

        (dice, op)
    }
}

impl Operation for Index {
    fn call(&self, values: &[Key]) -> Key {
        values[self.0]
    }

    fn shift_indices(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<T, F, O> Operation for Map<T, F>
where
    T: Operation + Clone,
    F: Fn(Key) -> O + Clone,
    O: Into<Key>,
{
    fn call(&self, values: &[Key]) -> Key {
        self.1(self.0.call(values)).into()
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for Negate<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
        Key::from(match self.0.call(values) {
            0 => 1,
            _ => 0,
        })
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for Eq<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
        let v = self.0.call(values);
        Key::from(self.1.contains(&v))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for Cmp<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values).cmp(&self.1) as Key
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    F: Fn(&[Key]) -> O + Clone,
    O: Into<Key>,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<L, R, F, O> Operation for FoldTwo<L, R, F>
where
    L: Operation + Clone,
    R: Operation + Clone,
    F: Fn(Key, Key) -> O + Clone,
    O: Into<Key>,
{
    fn call(&self, values: &[Key]) -> Key {
        self.2(self.0.call(values), self.1.call(values)).into()
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<T1, T2, T3, F, O> Operation for FoldThree<T1, T2, T3, F>
where
    T1: Operation + Clone,
    T2: Operation + Clone,
    T3: Operation + Clone,
    F: Fn(Key, Key, Key) -> O + Clone,
    O: Into<Key>,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<F, O> Operation for DynFold<F>
where
    F: Fn(&[Key]) -> O + Clone,
    O: Into<Key>,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl DynFoldBuilder {
    #[must_use]
    pub fn push<TR, R>(mut self, expr: TR) -> Self
    where
        TR: Expr<Op = R>,
        R: Operation + Clone + 'static,
    {
        let mut expr = expr.into_composite();
        self.0.extend(expr.dice);
        expr.op.shift_indices(self.0.len());
        self.1.push(expr.op.into_ptr());
        self
    }

    #[must_use]
    pub fn extend<I, TR, R>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = TR>,
        TR: Expr<Op = R>,
        R: Operation + Clone + 'static,
    {
        for i in iter {
            let mut i = i.into_composite();
            i.op.shift_indices(self.0.len());
            self.0.extend(i.dice);
            self.1.push(i.op.into_ptr());
        }
        self
    }

    #[must_use]
    pub fn build<F>(self, op: F) -> Composite<DynFold<F>>
    where
        F: Fn(&[Key]) -> Key + Clone + 'static,
    {
        Composite {
            dice: self.0,
            op: DynFold(self.1, op),
        }
    }
}

impl<T> Operation for Sum<T>
where
    T: Operation + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    fn call(&self, values: &[Key]) -> Key {
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
    F: Fn(Key) -> bool + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<T, F> Operation for All<T, F>
where
    T: Operation + Clone,
    F: Fn(Key) -> bool + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<F, C, L, R> Operation for Branch<F, C, L, R>
where
    F: Fn(Key) -> bool + Clone,
    C: Operation + Clone,
    L: Operation + Clone,
    R: Operation + Clone,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl Operation for Boxed {
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
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
}

impl Expr for Die {
    type Op = Index;

    fn into_composite(self) -> Composite<Self::Op> {
        Composite {
            dice: vec![self],
            op: Index(0),
        }
    }
}
