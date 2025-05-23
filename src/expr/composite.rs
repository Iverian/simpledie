use std::fmt::Debug;

use super::Operation;
use crate::util::{BigUint, DefaultKey, DieList, Value};
use crate::{Die, Key};

#[derive(Clone, Debug)]
pub struct Composite<T, K = DefaultKey>
where
    K: Key,
    T: Operation<K>,
{
    pub dice: DieList<K>,
    pub op: T,
}

impl<T, K> Composite<T, K>
where
    T: Operation<K>,
    K: Key,
{
    pub fn eval(self) -> Die<T::Output> {
        Die::<K>::eval(self.dice, move |x| self.op.call(x))
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

    pub(crate) fn explode(self, size: usize) -> (DieList<K>, Vec<T>) {
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

impl<K, R, T> From<Composite<T, K>> for Die<R>
where
    K: Key,
    R: Key,
    T: Operation<K, Output = R>,
{
    fn from(value: Composite<T, K>) -> Self {
        value.eval()
    }
}
