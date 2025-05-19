use std::fmt::Debug;

use super::Operation;
use crate::approx::Approx;
use crate::util::{BigUint, DieList, OverflowResult, Value};
use crate::Die;

#[derive(Clone, Debug)]
pub struct Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    pub(crate) dice: DieList,
    pub(crate) op: T,
}

impl<T> Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    pub fn eval(self) -> Die {
        Die::eval(self.dice, move |x| self.op.call(x))
    }

    pub fn eval_exact(self) -> OverflowResult<Die> {
        Die::eval_exact(self.dice, move |x| self.op.call(x))
    }

    pub fn eval_approx(self, approx: Approx) -> Die {
        Die::eval_approx(approx, &self.dice, move |x| self.op.call(x))
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

    pub(crate) fn explode(self, size: usize) -> (DieList, Vec<T>) {
        assert!(size != 0, "explode size cannot be zero");
        if size == 1 {
            return (self.dice, vec![self.op]);
        }

        let mut src = self;
        let mut dice = Vec::with_capacity(size * src.dice.len());
        let mut op = Vec::with_capacity(size);

        for _ in 1..size {
            let mut item = src.op.clone();
            item.shift_indices(dice.len());
            dice.extend(src.dice.iter().cloned());
            op.push(item);
        }

        src.op.shift_indices(dice.len());
        dice.extend(src.dice);
        op.push(src.op);

        (dice, op)
    }
}
