use std::fmt::Debug;

use super::Operation;
use crate::approx::Approx;
use crate::util::{BigUint, DieList, OverflowResult, Value};
use crate::{Die, EvalStrategy};

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
    pub fn eval_with_strategy(self, strategy: EvalStrategy) -> OverflowResult<Die> {
        Die::eval_with_strategy(strategy, self.dice, move |x| self.op.call(x))
    }

    pub fn eval(self) -> Die {
        Die::eval(self.dice, move |x| self.op.call(x))
    }

    pub fn eval_exact(self) -> OverflowResult<Die> {
        Die::eval_exact(self.dice, move |x| self.op.call(x))
    }

    pub fn eval_approx(self, approx: Approx) -> Die {
        Die::eval_approx(approx, self.dice, move |x| self.op.call(x))
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
