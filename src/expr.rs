use std::cmp::Ordering;

use crate::approx::Approx;
use crate::die::{Die, OverflowResult};
use crate::util::{cell, BigUint, Cell, DieList, Key, Value};

type OpPtr = Cell<dyn Operation + 'static>;

#[derive(Clone, Debug)]
pub struct Expr<T = Index>
where
    T: Operation,
{
    dice: DieList,
    op: T,
}

#[derive(Clone, Copy, Debug)]
pub struct Index(usize);

pub struct Map<T, F>(T, F);

#[derive(Clone, Copy, Debug)]
pub struct Negate<T>(T);

#[derive(Clone, Copy, Debug)]
pub struct AddKey<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct SubKey<T>(T, Key);

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
pub struct CombineTwo<L, R, F>(L, R, F);

#[derive(Clone, Copy, Debug)]
pub struct CombineThree<T1, T2, T3, F>(T1, T2, T3, F);

#[derive(Clone)]
pub struct Combine<F>(Vec<OpPtr>, F);

#[derive(Default)]
pub struct CombineBuilder(DieList, Vec<OpPtr>);

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
pub struct Erased(Cell<dyn Operation + 'static>);

pub trait Operation {
    fn call(&self, values: &[Key]) -> Key;
    fn shift_indices(&mut self, value: usize);
}

pub trait IntoExpr {
    type Op: Operation + 'static;

    fn into_expr(self) -> Expr<Self::Op>;
}

#[allow(clippy::module_name_repetitions)]
pub trait ExprExt: IntoExpr + Sized {
    fn eval(self) -> Die {
        self.into_expr().eval()
    }

    fn try_eval(self) -> OverflowResult<Die> {
        self.into_expr().try_eval()
    }

    fn approx_eval(self, approx: Approx) -> Die {
        self.into_expr().approx_eval(approx)
    }

    fn map<F>(self, op: F) -> Expr<Map<Self::Op, F>>
    where
        F: Fn(Key) -> Key + 'static,
    {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Map(me.op, op),
        }
    }

    fn neg(self) -> Expr<Negate<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Negate(me.op),
        }
    }

    fn kadd(self, rhs: Key) -> Expr<AddKey<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: AddKey(me.op, rhs),
        }
    }

    fn ksub(self, rhs: Key) -> Expr<SubKey<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: SubKey(me.op, rhs),
        }
    }

    fn kmul(self, rhs: Key) -> Expr<MulKey<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: MulKey(me.op, rhs),
        }
    }

    fn kdiv(self, rhs: Key) -> Expr<DivKey<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: DivKey(me.op, rhs),
        }
    }

    fn not(self) -> Expr<Not<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Not(me.op),
        }
    }

    fn contains(self, rhs: Vec<Key>) -> Expr<Eq<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Eq(me.op, rhs),
        }
    }

    fn eq(self, rhs: Key) -> Expr<Eq<Self::Op>> {
        self.contains(vec![rhs])
    }

    fn neq(self, rhs: Key) -> Expr<Not<Eq<Self::Op>>> {
        self.eq(rhs).not()
    }

    fn cmp(self, rhs: Key) -> Expr<Cmp<Self::Op>> {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Cmp(me.op, rhs),
        }
    }

    fn lt(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs).eq(Ordering::Less as Key)
    }

    fn le(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs)
            .contains(vec![Ordering::Less as Key, Ordering::Equal as Key])
    }

    fn gt(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs).eq(Ordering::Greater as Key)
    }

    fn ge(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>> {
        self.cmp(rhs)
            .contains(vec![Ordering::Greater as Key, Ordering::Equal as Key])
    }

    fn add<T>(self, rhs: T) -> Expr<Add<Self::Op, T::Op>>
    where
        T: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: Add(me.op, rhs.op),
        }
    }

    fn sub<T>(self, rhs: T) -> Expr<Add<Self::Op, Negate<T::Op>>>
    where
        T: IntoExpr,
    {
        self.add(rhs.into_expr().neg())
    }

    fn mul<T>(self, rhs: T) -> Expr<Mul<Self::Op, T::Op>>
    where
        T: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: Mul(me.op, rhs.op),
        }
    }

    fn div<T>(self, rhs: T) -> Expr<Div<Self::Op, T::Op>>
    where
        T: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: Div(me.op, rhs.op),
        }
    }

    fn min<T>(self, rhs: T) -> Expr<Min<Self::Op, T::Op>>
    where
        T: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: Min(me.op, rhs.op),
        }
    }

    fn max<T>(self, rhs: T) -> Expr<Max<Self::Op, T::Op>>
    where
        T: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: Max(me.op, rhs.op),
        }
    }

    fn fold<F>(self, size: usize, op: F) -> Expr<Fold<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(&[Key]) -> Key + 'static,
    {
        let (dice, expr) = self.into_expr().explode(size);
        Expr {
            dice,
            op: Fold(expr, op),
        }
    }

    fn combine_two<T, F>(self, rhs: T, op: F) -> Expr<CombineTwo<Self::Op, T::Op, F>>
    where
        T: IntoExpr,
        F: Fn(Key, Key) -> Key + 'static,
    {
        let mut me = self.into_expr();
        let mut rhs = rhs.into_expr();
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);
        Expr {
            dice: me.dice,
            op: CombineTwo(me.op, rhs.op, op),
        }
    }

    #[allow(clippy::type_complexity)]
    fn combine_three<T1, T2, F>(
        self,
        f: T1,
        s: T2,
        op: F,
    ) -> Expr<CombineThree<Self::Op, T1::Op, T2::Op, F>>
    where
        T1: IntoExpr,
        T2: IntoExpr,
        F: Fn(Key, Key, Key) -> Key + 'static,
    {
        let mut me = self.into_expr();
        let mut f = f.into_expr();
        let mut s = s.into_expr();
        f.op.shift_indices(me.dice.len());
        me.dice.extend(f.dice);
        s.op.shift_indices(me.dice.len());
        me.dice.extend(s.dice);
        Expr {
            dice: me.dice,
            op: CombineThree(me.op, f.op, s.op, op),
        }
    }

    fn combine(self) -> CombineBuilder
    where
        Self::Op: 'static,
    {
        let me = self.into_expr();
        CombineBuilder(me.dice, vec![cell(me.op)])
    }

    fn sum(self, size: usize) -> Expr<Sum<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr { dice, op: Sum(op) }
    }

    fn product(self, size: usize) -> Expr<Product<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr {
            dice,
            op: Product(op),
        }
    }

    fn min_of(self, size: usize) -> Expr<MinOf<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr {
            dice,
            op: MinOf(op),
        }
    }

    fn max_of(self, size: usize) -> Expr<MaxOf<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr {
            dice,
            op: MaxOf(op),
        }
    }

    fn any<F>(self, size: usize, pred: F) -> Expr<Any<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(Key) -> bool + 'static,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr {
            dice,
            op: Any(op, pred),
        }
    }

    fn all<F>(self, size: usize, pred: F) -> Expr<All<Self::Op, F>>
    where
        Self::Op: Clone,
        F: Fn(Key) -> bool + 'static,
    {
        let (dice, op) = self.into_expr().explode(size);
        Expr {
            dice,
            op: All(op, pred),
        }
    }

    #[allow(clippy::type_complexity)]
    fn branch<F, L, R>(self, pred: F, lhs: L, rhs: R) -> Expr<Branch<F, Self::Op, L::Op, R::Op>>
    where
        F: Fn(Key) -> bool,
        L: IntoExpr,
        R: IntoExpr,
    {
        let mut me = self.into_expr();
        let mut lhs = lhs.into_expr();
        let mut rhs = rhs.into_expr();

        lhs.op.shift_indices(me.dice.len());
        me.dice.extend(lhs.dice);
        rhs.op.shift_indices(me.dice.len());
        me.dice.extend(rhs.dice);

        Expr {
            dice: me.dice,
            op: Branch(pred, me.op, lhs.op, rhs.op),
        }
    }

    fn erase(self) -> Expr<Erased>
    where
        Self::Op: 'static,
    {
        let me = self.into_expr();
        Expr {
            dice: me.dice,
            op: Erased(cell(me.op)),
        }
    }
}

impl Expr {
    #[must_use]
    pub fn combine() -> CombineBuilder {
        CombineBuilder::default()
    }

    #[must_use]
    pub fn fold<F, I, E>(iter: I, op: F) -> Expr<Fold<E::Op, F>>
    where
        F: Fn(&[Key]) -> Key,
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: Fold(items, op),
        }
    }

    pub fn sum<I, E>(iter: I) -> Expr<Sum<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: Sum(items),
        }
    }

    pub fn product<I, E>(iter: I) -> Expr<Product<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: Product(items),
        }
    }

    pub fn min_of<I, E>(iter: I) -> Expr<MinOf<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: MinOf(items),
        }
    }

    pub fn max_of<I, E>(iter: I) -> Expr<MaxOf<E::Op>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: MaxOf(items),
        }
    }

    pub fn any<I, E, F>(iter: I, pred: F) -> Expr<Any<E::Op, F>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
        F: Fn(Key) -> bool,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: Any(items, pred),
        }
    }

    pub fn all<I, E, F>(iter: I, pred: F) -> Expr<All<E::Op, F>>
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
        F: Fn(Key) -> bool,
    {
        let (dice, items) = Self::parts(iter);
        Expr {
            dice,
            op: All(items, pred),
        }
    }

    fn parts<I, E>(iter: I) -> (Vec<Die>, Vec<E::Op>)
    where
        I: IntoIterator<Item = E>,
        E: IntoExpr,
    {
        let iter = iter.into_iter();
        let (l, u) = iter.size_hint();
        let s = u.unwrap_or(l);
        let mut dice = Vec::with_capacity(s);
        let mut items = Vec::with_capacity(s);
        for i in iter {
            let mut i = i.into_expr();
            i.op.shift_indices(dice.len());
            dice.extend(i.dice);
            items.push(i.op);
        }
        (dice, items)
    }
}

impl<T> Expr<T>
where
    T: Operation,
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

impl<T> Expr<T>
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

impl<T, F> Operation for Map<T, F>
where
    T: Operation,
    F: Fn(Key) -> Key,
{
    fn call(&self, values: &[Key]) -> Key {
        self.1(self.0.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T: Operation> Operation for Negate<T> {
    fn call(&self, values: &[Key]) -> Key {
        -self.0.call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for AddKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) + self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for SubKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) - self.1
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
    }
}

impl<T> Operation for MulKey<T>
where
    T: Operation,
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
    T: Operation,
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
    T: Operation,
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
    T: Operation,
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
    T: Operation,
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
    L: Operation,
    R: Operation,
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
    L: Operation,
    R: Operation,
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
    L: Operation,
    R: Operation,
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
    L: Operation,
    R: Operation,
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
    L: Operation,
    R: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values).max(self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<T, F> Operation for Fold<T, F>
where
    T: Operation,
    F: Fn(&[Key]) -> Key,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<L, R, F> Operation for CombineTwo<L, R, F>
where
    L: Operation,
    R: Operation,
    F: Fn(Key, Key) -> Key + 'static,
{
    fn call(&self, values: &[Key]) -> Key {
        self.2(self.0.call(values), self.1.call(values))
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.shift_indices(value);
        self.1.shift_indices(value);
    }
}

impl<T1, T2, T3, F> Operation for CombineThree<T1, T2, T3, F>
where
    T1: Operation,
    T2: Operation,
    T3: Operation,
    F: Fn(Key, Key, Key) -> Key + 'static,
{
    fn call(&self, values: &[Key]) -> Key {
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

impl<F> Operation for Combine<F>
where
    F: Fn(&[Key]) -> Key,
{
    fn call(&self, values: &[Key]) -> Key {
        self.1(
            self.0
                .iter()
                .map(|x| x.borrow().call(values))
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn shift_indices(&mut self, value: usize) {
        for x in &mut self.0 {
            x.borrow_mut().shift_indices(value);
        }
    }
}

impl CombineBuilder {
    #[must_use]
    pub fn push<TR, R>(mut self, expr: TR) -> Self
    where
        TR: IntoExpr<Op = R>,
        R: Operation + 'static,
    {
        let mut expr = expr.into_expr();
        self.0.extend(expr.dice);
        expr.op.shift_indices(self.0.len());
        self.1.push(cell(expr.op));
        self
    }

    #[must_use]
    pub fn extend<I, TR, R>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = TR>,
        TR: IntoExpr<Op = R>,
        R: Operation + 'static,
    {
        for i in iter {
            let mut i = i.into_expr();
            i.op.shift_indices(self.0.len());
            self.0.extend(i.dice);
            self.1.push(cell(i.op));
        }
        self
    }

    #[must_use]
    pub fn build<F>(self, op: F) -> Expr<Combine<F>>
    where
        F: Fn(&[Key]) -> Key + 'static,
    {
        Expr {
            dice: self.0,
            op: Combine(self.1, op),
        }
    }
}

impl<T> Operation for Sum<T>
where
    T: Operation,
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
    T: Operation,
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
    T: Operation,
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
    T: Operation,
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
    T: Operation,
    F: Fn(Key) -> bool,
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
    T: Operation,
    F: Fn(Key) -> bool,
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
    F: Fn(Key) -> bool,
    C: Operation,
    L: Operation,
    R: Operation,
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

impl Operation for Erased {
    fn call(&self, values: &[Key]) -> Key {
        self.0.borrow().call(values)
    }

    fn shift_indices(&mut self, value: usize) {
        self.0.borrow_mut().shift_indices(value);
    }
}

impl<T> IntoExpr for Expr<T>
where
    T: Operation + 'static,
{
    type Op = T;

    fn into_expr(self) -> Expr<Self::Op> {
        self
    }
}

impl IntoExpr for Die {
    type Op = Index;

    fn into_expr(self) -> Expr<Self::Op> {
        Expr {
            dice: vec![self],
            op: Index(0),
        }
    }
}

impl<T> ExprExt for T where T: IntoExpr + Sized {}
