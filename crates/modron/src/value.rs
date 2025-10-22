use std::cmp::Ordering;
use std::fmt::Debug;

pub type DefaultValue = i32;
pub type ComputedValue = i64;

pub trait Value: Sized + Send + Sync + Debug + Clone + PartialEq + Eq + PartialOrd + Ord {}

pub trait OrderedValue: PartialOrd + Ord + Value {}

pub trait ComputableValue: Value {
    fn compute(&self) -> ComputedValue;

    fn compute_f64(&self) -> f64 {
        self.compute() as f64
    }
}

macro_rules! impl_computable_value_from {
    ($typ:ty) => {
        impl $crate::value::ComputableValue for $typ {
            fn compute(&self) -> ComputedValue {
                ComputedValue::from(*self)
            }
        }
    };
}

impl<T> Value for T where T: Sized + Send + Sync + Debug + Clone + PartialEq + Eq + PartialEq + Ord {}

impl<T> OrderedValue for T where T: Value + PartialOrd + Ord {}

impl ComputableValue for bool {
    fn compute(&self) -> ComputedValue {
        if *self {
            1
        } else {
            0
        }
    }
}

impl ComputableValue for Ordering {
    fn compute(&self) -> ComputedValue {
        match self {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

impl_computable_value_from!(u8);
impl_computable_value_from!(u16);
impl_computable_value_from!(u32);
impl_computable_value_from!(i8);
impl_computable_value_from!(i16);
impl_computable_value_from!(i32);
