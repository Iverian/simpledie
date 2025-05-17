use std::cmp::Ordering;

use crate::die::Die;
use crate::util::{BigUint, Key, Rc, Value};

pub type BoxMapFn = Box<dyn Fn(Key) -> Key + 'static>;
pub type BoxCombineFn = Box<dyn Fn(&[Key]) -> Key + 'static>;

#[derive(Clone, Debug)]
pub struct Evaluation<T>
where
    T: Expression,
{
    dice: Vec<Rc<Die>>,
    expr: T,
}

#[derive(Clone, Copy, Debug)]
pub struct IdentityExpr(usize);

pub struct MapExpr<T>(T, BoxMapFn);

#[derive(Clone, Copy, Debug)]
pub struct NegateExpr<T>(T);

#[derive(Clone, Copy, Debug)]
pub struct AddKeyExpr<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct SubKeyExpr<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct MulKeyExpr<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct DivKeyExpr<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct NotExpr<T>(T);

#[derive(Clone, Debug)]
pub struct EqExpr<T>(T, Vec<Key>);

#[derive(Clone, Copy, Debug)]
pub struct CompareExpr<T>(T, Key);

#[derive(Clone, Copy, Debug)]
pub struct AddExpr<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct MulExpr<L, R>(L, R);

#[derive(Clone, Copy, Debug)]
pub struct DivExpr<L, R>(L, R);

pub struct CombineExpr<T>(Vec<T>, BoxCombineFn);

#[derive(Clone, Debug)]
pub struct SumExpr<T>(Vec<T>);

#[derive(Clone, Debug)]
pub struct ProductExpr<T>(Vec<T>);

#[derive(Clone, Debug, Copy)]
pub struct BranchExpr<C, L, R>(C, L, R);

pub trait Expression {
    fn eval(&self, values: &[Key]) -> Key;
    fn shift_identity(&mut self, value: usize);
}

pub trait EvaluationExt {
    type Expr: Expression;

    fn map<F>(self, op: F) -> Evaluation<MapExpr<Self::Expr>>
    where
        F: Fn(Key) -> Key + 'static;

    fn neg(self) -> Evaluation<NegateExpr<Self::Expr>>;

    fn kadd(self, rhs: Key) -> Evaluation<AddKeyExpr<Self::Expr>>;

    fn ksub(self, rhs: Key) -> Evaluation<SubKeyExpr<Self::Expr>>;

    fn kmul(self, rhs: Key) -> Evaluation<MulKeyExpr<Self::Expr>>;

    fn kdiv(self, rhs: Key) -> Evaluation<DivKeyExpr<Self::Expr>>;

    fn not(self) -> Evaluation<NotExpr<Self::Expr>>;

    fn any(self, rhs: Vec<Key>) -> Evaluation<EqExpr<Self::Expr>>;

    fn eq(self, rhs: Key) -> Evaluation<EqExpr<Self::Expr>>
    where
        Self: Sized,
    {
        self.any(vec![rhs])
    }

    fn neq(self, rhs: Key) -> Evaluation<NotExpr<EqExpr<Self::Expr>>>
    where
        Self: Sized,
    {
        self.eq(rhs).not()
    }

    fn cmp(self, rhs: Key) -> Evaluation<CompareExpr<Self::Expr>>;

    fn lt(self, rhs: Key) -> Evaluation<EqExpr<CompareExpr<Self::Expr>>>
    where
        Self: Sized,
    {
        self.cmp(rhs).eq(Ordering::Less as Key)
    }

    fn le(self, rhs: Key) -> Evaluation<EqExpr<CompareExpr<Self::Expr>>>
    where
        Self: Sized,
    {
        self.cmp(rhs)
            .any(vec![Ordering::Less as Key, Ordering::Equal as Key])
    }

    fn gt(self, rhs: Key) -> Evaluation<EqExpr<CompareExpr<Self::Expr>>>
    where
        Self: Sized,
    {
        self.cmp(rhs).eq(Ordering::Greater as Key)
    }

    fn ge(self, rhs: Key) -> Evaluation<EqExpr<CompareExpr<Self::Expr>>>
    where
        Self: Sized,
    {
        self.cmp(rhs)
            .any(vec![Ordering::Greater as Key, Ordering::Equal as Key])
    }

    fn add<R>(self, rhs: Evaluation<R>) -> Evaluation<AddExpr<Self::Expr, R>>
    where
        R: Expression;

    fn sub<R>(self, rhs: Evaluation<R>) -> Evaluation<AddExpr<Self::Expr, NegateExpr<R>>>
    where
        Self: Sized,
        R: Expression,
    {
        self.add(rhs.neg())
    }

    fn mul<R>(self, rhs: Evaluation<R>) -> Evaluation<MulExpr<Self::Expr, R>>
    where
        R: Expression;

    fn div<R>(self, rhs: Evaluation<R>) -> Evaluation<DivExpr<Self::Expr, R>>
    where
        R: Expression;

    fn combine<F>(self, size: usize, op: F) -> Evaluation<CombineExpr<Self::Expr>>
    where
        Self::Expr: Clone,
        F: Fn(&[Key]) -> Key + 'static;

    fn sum(self, size: usize) -> Evaluation<SumExpr<Self::Expr>>
    where
        Self::Expr: Clone;

    fn product(self, size: usize) -> Evaluation<ProductExpr<Self::Expr>>
    where
        Self::Expr: Clone;

    fn branch<L, R>(
        self,
        lhs: Evaluation<L>,
        rhs: Evaluation<R>,
    ) -> Evaluation<BranchExpr<Self::Expr, L, R>>
    where
        L: Expression,
        R: Expression;
}

impl<T> Evaluation<T>
where
    T: Expression,
{
    pub fn collect(self) -> Die {
        Die::combine(self.dice, move |x| self.expr.eval(x))
    }

    pub fn denom(self) -> BigUint {
        self.dice
            .iter()
            .map(|x| BigUint::from(x.denom()))
            .fold(BigUint::from(1 as Value), |acc, x| acc * x)
    }
}

impl<T> Evaluation<T>
where
    T: Expression + Clone,
{
    fn explode(self, size: usize) -> (Vec<Rc<Die>>, Vec<T>) {
        let m = self.dice.len();
        let mut dice = Vec::with_capacity(size * m);
        let mut expr = Vec::with_capacity(size);
        for i in 0..size {
            let mut e = self.expr.clone();
            e.shift_identity(i * m);
            expr.push(e);
            dice.extend(self.dice.iter().cloned());
        }
        (dice, expr)
    }
}

impl From<Die> for Evaluation<IdentityExpr> {
    fn from(value: Die) -> Self {
        Evaluation {
            dice: vec![Rc::new(value)],
            expr: IdentityExpr(0),
        }
    }
}

impl<T> From<Evaluation<T>> for Die
where
    T: Expression,
{
    fn from(value: Evaluation<T>) -> Self {
        value.collect()
    }
}

impl Expression for IdentityExpr {
    fn eval(&self, values: &[Key]) -> Key {
        values[self.0]
    }

    fn shift_identity(&mut self, value: usize) {
        self.0 += value;
    }
}

impl<T> Expression for MapExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.1(self.0.eval(values))
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T: Expression> Expression for NegateExpr<T> {
    fn eval(&self, values: &[Key]) -> Key {
        -self.0.eval(values)
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for AddKeyExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.eval(values) + self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for SubKeyExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.eval(values) - self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for MulKeyExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.eval(values) * self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for DivKeyExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.eval(values) / self.1
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for NotExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        Key::from(match self.0.eval(values) {
            0 => 1,
            _ => 0,
        })
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for EqExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        let v = self.0.eval(values);
        Key::from(self.1.contains(&v))
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<T> Expression for CompareExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.eval(values).cmp(&self.1) as Key
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
    }
}

impl<L, R> Expression for AddExpr<L, R>
where
    L: Expression,
    R: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        let lhs = self.0.eval(values);
        let rhs = self.1.eval(values);
        lhs + rhs
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
    }
}

impl<L, R> Expression for MulExpr<L, R>
where
    L: Expression,
    R: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        let lhs = self.0.eval(values);
        let rhs = self.1.eval(values);
        lhs * rhs
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
    }
}

impl<L, R> Expression for DivExpr<L, R>
where
    L: Expression,
    R: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        let lhs = self.0.eval(values);
        let rhs = self.1.eval(values);
        lhs / rhs
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
    }
}

impl<T> Expression for CombineExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        let x: Vec<_> = self.0.iter().map(|x| x.eval(values)).collect();
        self.1(x.as_slice())
    }

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
        }
    }
}

impl<T> Expression for SumExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.iter().map(|x| x.eval(values)).sum()
    }

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
        }
    }
}

impl<T> Expression for ProductExpr<T>
where
    T: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        self.0.iter().map(|x| x.eval(values)).product()
    }

    fn shift_identity(&mut self, value: usize) {
        for x in &mut self.0 {
            x.shift_identity(value);
        }
    }
}

impl<C, L, R> Expression for BranchExpr<C, L, R>
where
    C: Expression,
    L: Expression,
    R: Expression,
{
    fn eval(&self, values: &[Key]) -> Key {
        if self.0.eval(values) != 0 {
            self.1.eval(values)
        } else {
            self.2.eval(values)
        }
    }

    fn shift_identity(&mut self, value: usize) {
        self.0.shift_identity(value);
        self.1.shift_identity(value);
        self.2.shift_identity(value);
    }
}

impl<E> EvaluationExt for Evaluation<E>
where
    E: Expression,
{
    type Expr = E;

    fn map<F>(self, op: F) -> Evaluation<MapExpr<Self::Expr>>
    where
        F: Fn(Key) -> Key + 'static,
    {
        Evaluation {
            dice: self.dice,
            expr: MapExpr(self.expr, Box::new(op)),
        }
    }

    fn neg(self) -> Evaluation<NegateExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: NegateExpr(self.expr),
        }
    }

    fn kadd(self, rhs: Key) -> Evaluation<AddKeyExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: AddKeyExpr(self.expr, rhs),
        }
    }

    fn ksub(self, rhs: Key) -> Evaluation<SubKeyExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: SubKeyExpr(self.expr, rhs),
        }
    }

    fn kmul(self, rhs: Key) -> Evaluation<MulKeyExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: MulKeyExpr(self.expr, rhs),
        }
    }

    fn kdiv(self, rhs: Key) -> Evaluation<DivKeyExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: DivKeyExpr(self.expr, rhs),
        }
    }

    fn not(self) -> Evaluation<NotExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: NotExpr(self.expr),
        }
    }

    fn any(self, rhs: Vec<Key>) -> Evaluation<EqExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: EqExpr(self.expr, rhs),
        }
    }

    fn cmp(self, rhs: Key) -> Evaluation<CompareExpr<Self::Expr>> {
        Evaluation {
            dice: self.dice,
            expr: CompareExpr(self.expr, rhs),
        }
    }

    fn add<R>(mut self, mut rhs: Evaluation<R>) -> Evaluation<AddExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        rhs.expr.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Evaluation {
            dice: self.dice,
            expr: AddExpr(self.expr, rhs.expr),
        }
    }

    fn mul<R>(mut self, mut rhs: Evaluation<R>) -> Evaluation<MulExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        rhs.expr.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Evaluation {
            dice: self.dice,
            expr: MulExpr(self.expr, rhs.expr),
        }
    }

    fn div<R>(mut self, mut rhs: Evaluation<R>) -> Evaluation<DivExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        rhs.expr.shift_identity(self.dice.len());
        self.dice.extend(rhs.dice);
        Evaluation {
            dice: self.dice,
            expr: DivExpr(self.expr, rhs.expr),
        }
    }

    fn combine<F>(self, size: usize, op: F) -> Evaluation<CombineExpr<Self::Expr>>
    where
        Self::Expr: Clone,
        F: Fn(&[Key]) -> Key + 'static,
    {
        let (dice, expr) = self.explode(size);
        Evaluation {
            dice,
            expr: CombineExpr(expr, Box::new(op)),
        }
    }

    fn sum(self, size: usize) -> Evaluation<SumExpr<Self::Expr>>
    where
        Self::Expr: Clone,
    {
        let (dice, expr) = self.explode(size);
        Evaluation {
            dice,
            expr: SumExpr(expr),
        }
    }

    fn product(self, size: usize) -> Evaluation<ProductExpr<Self::Expr>>
    where
        Self::Expr: Clone,
    {
        let (dice, expr) = self.explode(size);
        Evaluation {
            dice,
            expr: ProductExpr(expr),
        }
    }

    fn branch<L, R>(
        mut self,
        mut lhs: Evaluation<L>,
        mut rhs: Evaluation<R>,
    ) -> Evaluation<BranchExpr<Self::Expr, L, R>>
    where
        L: Expression,
        R: Expression,
    {
        let sl = self.dice.len();
        let ll = lhs.dice.len();

        self.dice.extend(lhs.dice);
        self.dice.extend(rhs.dice);

        lhs.expr.shift_identity(sl);
        rhs.expr.shift_identity(sl + ll);

        Evaluation {
            dice: self.dice,
            expr: BranchExpr(self.expr, lhs.expr, rhs.expr),
        }
    }
}

impl EvaluationExt for Die {
    type Expr = IdentityExpr;

    fn map<F>(self, op: F) -> Evaluation<MapExpr<Self::Expr>>
    where
        F: Fn(Key) -> Key + 'static,
    {
        Evaluation::from(self).map(op)
    }

    fn neg(self) -> Evaluation<NegateExpr<Self::Expr>> {
        Evaluation::from(self).neg()
    }

    fn kadd(self, rhs: Key) -> Evaluation<AddKeyExpr<Self::Expr>> {
        Evaluation::from(self).kadd(rhs)
    }

    fn ksub(self, rhs: Key) -> Evaluation<SubKeyExpr<Self::Expr>> {
        Evaluation::from(self).ksub(rhs)
    }

    fn kmul(self, rhs: Key) -> Evaluation<MulKeyExpr<Self::Expr>> {
        Evaluation::from(self).kmul(rhs)
    }

    fn kdiv(self, rhs: Key) -> Evaluation<DivKeyExpr<Self::Expr>> {
        Evaluation::from(self).kdiv(rhs)
    }

    fn not(self) -> Evaluation<NotExpr<Self::Expr>> {
        Evaluation::from(self).not()
    }

    fn any(self, rhs: Vec<Key>) -> Evaluation<EqExpr<Self::Expr>> {
        Evaluation::from(self).any(rhs)
    }

    fn cmp(self, rhs: Key) -> Evaluation<CompareExpr<Self::Expr>> {
        Evaluation::from(self).cmp(rhs)
    }

    fn add<R>(self, rhs: Evaluation<R>) -> Evaluation<AddExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        Evaluation::from(self).add(rhs)
    }

    fn mul<R>(self, rhs: Evaluation<R>) -> Evaluation<MulExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        Evaluation::from(self).mul(rhs)
    }

    fn div<R>(self, rhs: Evaluation<R>) -> Evaluation<DivExpr<Self::Expr, R>>
    where
        R: Expression,
    {
        Evaluation::from(self).div(rhs)
    }

    fn combine<F>(self, size: usize, op: F) -> Evaluation<CombineExpr<Self::Expr>>
    where
        Self::Expr: Clone,
        F: Fn(&[Key]) -> Key + 'static,
    {
        Evaluation::from(self).combine(size, op)
    }

    fn sum(self, size: usize) -> Evaluation<SumExpr<Self::Expr>>
    where
        Self::Expr: Clone,
    {
        Evaluation::from(self).sum(size)
    }

    fn product(self, size: usize) -> Evaluation<ProductExpr<Self::Expr>>
    where
        Self::Expr: Clone,
    {
        Evaluation::from(self).product(size)
    }

    fn branch<L, R>(
        self,
        lhs: Evaluation<L>,
        rhs: Evaluation<R>,
    ) -> Evaluation<BranchExpr<Self::Expr, L, R>>
    where
        L: Expression,
        R: Expression,
    {
        Evaluation::from(self).branch(lhs, rhs)
    }
}
