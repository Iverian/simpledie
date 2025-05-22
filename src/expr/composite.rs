use std::fmt::Debug;

use super::Operation;
use crate::util::{BigUint, DieList, OverflowResult, Value};
use crate::Die;

#[derive(Clone, Debug)]
pub struct Composite<K, T>
where
    K: Clone + Copy + Ord + Debug,
    T: Operation<K> + Clone + Debug + Send + 'static,
{
    pub dice: DieList<K>,
    pub op: T,
}

impl<K, T> Composite<K, T>
where
    K: Clone + Copy + Ord + Debug,
    T: Operation<K> + Clone + Debug + Send + 'static,
{
    pub fn eval_exact(self) -> OverflowResult<Die<T::Output>> {
        Die::eval_exact(self.dice, move |x| self.op.call(x))
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

impl<K, T> Composite<K, T>
where
    K: Clone + Copy + Ord + Debug,
    T: Operation<K> + Clone + Debug + Send + 'static,
    T::Output: TryInto<f64>,
{
    pub fn eval(self) -> Result<Die<T::Output>, <T::Output as TryInto<f64>>::Error> {
        Die::eval(self.dice, move |x| self.op.call(x))
    }
}
