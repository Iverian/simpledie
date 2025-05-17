use std::cmp::Ordering;

use crate::approx::Approx;
use crate::die::{Die, OverflowResult};
use crate::util::{BigUint, Key, Rc, Value};

pub type BoxMapFn = Box<dyn Fn(Key) -> Key + 'static>;
pub type BoxCombineFn = Box<dyn Fn(&[Key]) -> Key + 'static>;

#[derive(Clone, Debug)]
pub struct Expr<T>
where
    T: Operation,
{
    dice: Vec<Rc<Die>>,
    op: T,
}

#[derive(Clone, Copy, Debug)]
pub struct Identity(usize);

pub struct MapExpr<T>(T, BoxMapFn);

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

pub struct Combine<T>(Vec<T>, BoxCombineFn);

#[derive(Clone, Debug)]
pub struct Sum<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct Product<T>(Vec<T>);

#[derive(Clone, Debug, Copy)]
pub struct Branch<C, L, R>(C, L, R);

pub trait Operation {
    fn call(&self, values: &[Key]) -> Key;
    fn shift_identity(&mut self, value: usize);
}

pub trait ExprExt {
    type Op: Operation;

    fn map<F>(self, op: F) -> Expr<MapExpr<Self::Op>>
    where
        F: Fn(Key) -> Key + 'static;

    fn neg(self) -> Expr<Negate<Self::Op>>;

    fn kadd(self, rhs: Key) -> Expr<AddKey<Self::Op>>;

    fn ksub(self, rhs: Key) -> Expr<SubKey<Self::Op>>;

    fn kmul(self, rhs: Key) -> Expr<MulKey<Self::Op>>;

    fn kdiv(self, rhs: Key) -> Expr<DivKey<Self::Op>>;

    fn not(self) -> Expr<Not<Self::Op>>;

    fn any(self, rhs: Vec<Key>) -> Expr<Eq<Self::Op>>;

    fn eq(self, rhs: Key) -> Expr<Eq<Self::Op>>
    where
        Self: Sized,
    {
        self.any(vec![rhs])
    }

    fn neq(self, rhs: Key) -> Expr<Not<Eq<Self::Op>>>
    where
        Self: Sized,
    {
        self.eq(rhs).not()
    }

    fn cmp(self, rhs: Key) -> Expr<Cmp<Self::Op>>;

    fn lt(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>>
    where
        Self: Sized,
    {
        self.cmp(rhs).eq(Ordering::Less as Key)
    }

    fn le(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>>
    where
        Self: Sized,
    {
        self.cmp(rhs)
            .any(vec![Ordering::Less as Key, Ordering::Equal as Key])
    }

    fn gt(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>>
    where
        Self: Sized,
    {
        self.cmp(rhs).eq(Ordering::Greater as Key)
    }

    fn ge(self, rhs: Key) -> Expr<Eq<Cmp<Self::Op>>>
    where
        Self: Sized,
    {
        self.cmp(rhs)
            .any(vec![Ordering::Greater as Key, Ordering::Equal as Key])
    }

    fn add<T, R>(self, rhs: T) -> Expr<Add<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation;

    fn sub<T, R>(self, rhs: T) -> Expr<Add<Self::Op, Negate<R>>>
    where
        Self: Sized,
        T: Into<Expr<R>>,
        R: Operation,
    {
        self.add(rhs.into().neg())
    }

    fn mul<T, R>(self, rhs: T) -> Expr<Mul<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation;

    fn div<T, R>(self, rhs: T) -> Expr<Div<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation;

    fn combine<F>(self, size: usize, op: F) -> Expr<Combine<Self::Op>>
    where
        Self::Op: Clone,
        F: Fn(&[Key]) -> Key + 'static;

    fn sum(self, size: usize) -> Expr<Sum<Self::Op>>
    where
        Self::Op: Clone;

    fn product(self, size: usize) -> Expr<Product<Self::Op>>
    where
        Self::Op: Clone;

    fn branch<TL, TR, L, R>(self, lhs: TL, rhs: TR) -> Expr<Branch<Self::Op, L, R>>
    where
        TL: Into<Expr<L>>,
        TR: Into<Expr<R>>,
        L: Operation,
        R: Operation;
}

impl<T> Expr<T>
where
    T: Operation,
{
    pub fn eval(self) -> Die {
        Die::combine(self.dice, move |x| self.op.call(x))
    }

    pub fn try_eval(self) -> OverflowResult<Die> {
        Die::try_combine(self.dice, move |x| self.op.call(x))
    }

    pub fn approx_eval(self, approx: Approx) -> Die {
        Die::combine_approx(approx, self.dice, move |x| self.op.call(x))
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
    fn explode(self, size: usize) -> (Vec<Rc<Die>>, Vec<T>) {
        let m = self.dice.len();
        let mut dice = Vec::with_capacity(size * m);
        let mut expr = Vec::with_capacity(size);
        for i in 0..size {
            let mut e = self.op.clone();
            e.shift_identity(i * m);
            expr.push(e);
            dice.extend(self.dice.iter().cloned());
        }
        (dice, expr)
    }
}

impl From<Die> for Expr<Identity> {
    fn from(value: Die) -> Self {
        Expr {
            dice: vec![Rc::new(value)],
            op: Identity(0),
        }
    }
}

impl<T> From<Expr<T>> for Die
where
    T: Operation,
{
    fn from(value: Expr<T>) -> Self {
        value.eval()
    }
}

impl Operation for Identity {
    fn call(&self, values: &[Key]) -> Key {
        values[self.0]
    }

    fn shift_identity(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<T> Operation for MapExpr<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.1(self.0.call(values))
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T: Operation> Operation for Negate<T> {
    fn call(&self, values: &[Key]) -> Key {
        -self.0.call(values)
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Operation for AddKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) + self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Operation for SubKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) - self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Operation for MulKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) * self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Operation for DivKey<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values) / self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Operation for Cmp<T>
where
    T: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        self.0.call(values).cmp(&self.1) as Key
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
    }
}

impl<T> Operation for Combine<T>
where
    T: Operation,
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

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
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

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
        }
    }
}

impl<C, L, R> Operation for Branch<C, L, R>
where
    C: Operation,
    L: Operation,
    R: Operation,
{
    fn call(&self, values: &[Key]) -> Key {
        if self.0.call(values) != 0 {
            self.1.call(values)
        } else {
            self.2.call(values)
        }
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
        self.2.shift_identity(value);
    }
}

impl<E> ExprExt for Expr<E>
where
    E: Operation,
{
    type Op = E;

    fn map<F>(self, op: F) -> Expr<MapExpr<Self::Op>>
    where
        F: Fn(Key) -> Key + 'static,
    {
        Expr {
            dice: self.dice,
            op: MapExpr(self.op, Box::new(op)),
        }
    }

    fn neg(self) -> Expr<Negate<Self::Op>> {
        Expr {
            dice: self.dice,
            op: Negate(self.op),
        }
    }

    fn kadd(self, rhs: Key) -> Expr<AddKey<Self::Op>> {
        Expr {
            dice: self.dice,
            op: AddKey(self.op, rhs),
        }
    }

    fn ksub(self, rhs: Key) -> Expr<SubKey<Self::Op>> {
        Expr {
            dice: self.dice,
            op: SubKey(self.op, rhs),
        }
    }

    fn kmul(self, rhs: Key) -> Expr<MulKey<Self::Op>> {
        Expr {
            dice: self.dice,
            op: MulKey(self.op, rhs),
        }
    }

    fn kdiv(self, rhs: Key) -> Expr<DivKey<Self::Op>> {
        Expr {
            dice: self.dice,
            op: DivKey(self.op, rhs),
        }
    }

    fn not(self) -> Expr<Not<Self::Op>> {
        Expr {
            dice: self.dice,
            op: Not(self.op),
        }
    }

    fn any(self, rhs: Vec<Key>) -> Expr<Eq<Self::Op>> {
        Expr {
            dice: self.dice,
            op: Eq(self.op, rhs),
        }
    }

    fn cmp(self, rhs: Key) -> Expr<Cmp<Self::Op>> {
        Expr {
            dice: self.dice,
            op: Cmp(self.op, rhs),
        }
    }

    fn add<T, R>(mut self, rhs: T) -> Expr<Add<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        let mut rhs = rhs.into();
        rhs.op.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Expr {
            dice: self.dice,
            op: Add(self.op, rhs.op),
        }
    }

    fn mul<T, R>(mut self, rhs: T) -> Expr<Mul<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        let mut rhs = rhs.into();
        rhs.op.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Expr {
            dice: self.dice,
            op: Mul(self.op, rhs.op),
        }
    }

    fn div<T, R>(mut self, rhs: T) -> Expr<Div<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        let mut rhs = rhs.into();
        rhs.op.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Expr {
            dice: self.dice,
            op: Div(self.op, rhs.op),
        }
    }

    fn combine<F>(self, size: usize, op: F) -> Expr<Combine<Self::Op>>
    where
        Self::Op: Clone,
        F: Fn(&[Key]) -> Key + 'static,
    {
        let (dice, expr) = self.explode(size);
        Expr {
            dice,
            op: Combine(expr, Box::new(op)),
        }
    }

    fn sum(self, size: usize) -> Expr<Sum<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, expr) = self.explode(size);
        Expr {
            dice,
            op: Sum(expr),
        }
    }

    fn product(self, size: usize) -> Expr<Product<Self::Op>>
    where
        Self::Op: Clone,
    {
        let (dice, expr) = self.explode(size);
        Expr {
            dice,
            op: Product(expr),
        }
    }

    fn branch<TL, TR, L, R>(mut self, lhs: TL, rhs: TR) -> Expr<Branch<Self::Op, L, R>>
    where
        TL: Into<Expr<L>>,
        TR: Into<Expr<R>>,
        L: Operation,
        R: Operation,
    {
        let mut lhs = lhs.into();
        let mut rhs = rhs.into();

        let sl = self.dice.len();
        let ll = lhs.dice.len();

        self.dice.extend(lhs.dice);
        self.dice.extend(rhs.dice);

        lhs.op.shift_identity(sl);
        rhs.op.shift_identity(sl + ll);

        Expr {
            dice: self.dice,
            op: Branch(self.op, lhs.op, rhs.op),
        }
    }
}

impl ExprExt for Die {
    type Op = Identity;

    fn map<F>(self, op: F) -> Expr<MapExpr<Self::Op>>
    where
        F: Fn(Key) -> Key + 'static,
    {
        Expr::from(self).map(op)
    }

    fn neg(self) -> Expr<Negate<Self::Op>> {
        Expr::from(self).neg()
    }

    fn kadd(self, rhs: Key) -> Expr<AddKey<Self::Op>> {
        Expr::from(self).kadd(rhs)
    }

    fn ksub(self, rhs: Key) -> Expr<SubKey<Self::Op>> {
        Expr::from(self).ksub(rhs)
    }

    fn kmul(self, rhs: Key) -> Expr<MulKey<Self::Op>> {
        Expr::from(self).kmul(rhs)
    }

    fn kdiv(self, rhs: Key) -> Expr<DivKey<Self::Op>> {
        Expr::from(self).kdiv(rhs)
    }

    fn not(self) -> Expr<Not<Self::Op>> {
        Expr::from(self).not()
    }

    fn any(self, rhs: Vec<Key>) -> Expr<Eq<Self::Op>> {
        Expr::from(self).any(rhs)
    }

    fn cmp(self, rhs: Key) -> Expr<Cmp<Self::Op>> {
        Expr::from(self).cmp(rhs)
    }

    fn add<T, R>(self, rhs: T) -> Expr<Add<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        Expr::from(self).add(rhs)
    }

    fn mul<T, R>(self, rhs: T) -> Expr<Mul<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        Expr::from(self).mul(rhs)
    }

    fn div<T, R>(self, rhs: T) -> Expr<Div<Self::Op, R>>
    where
        T: Into<Expr<R>>,
        R: Operation,
    {
        Expr::from(self).div(rhs)
    }

    fn combine<F>(self, size: usize, op: F) -> Expr<Combine<Self::Op>>
    where
        Self::Op: Clone,
        F: Fn(&[Key]) -> Key + 'static,
    {
        Expr::from(self).combine(size, op)
    }

    fn sum(self, size: usize) -> Expr<Sum<Self::Op>>
    where
        Self::Op: Clone,
    {
        Expr::from(self).sum(size)
    }

    fn product(self, size: usize) -> Expr<Product<Self::Op>>
    where
        Self::Op: Clone,
    {
        Expr::from(self).product(size)
    }

    fn branch<TL, TR, L, R>(self, lhs: TL, rhs: TR) -> Expr<Branch<Self::Op, L, R>>
    where
        TL: Into<Expr<L>>,
        TR: Into<Expr<R>>,
        L: Operation,
        R: Operation,
    {
        Expr::from(self).branch(lhs, rhs)
    }
}
