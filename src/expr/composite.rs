use std::fmt::Debug;

use super::Operation;
use crate::util::{BigUint, DieList, Value};
use crate::Die;

#[derive(Clone, Debug)]
pub struct Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    pub dice: DieList,
    pub op: T,
}

impl<T> Composite<T>
where
    T: Operation + Clone + Debug + Send + 'static,
{
    pub fn eval(self) -> Die {
        Die::eval(self.dice, move |x| self.op.call(x))
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
