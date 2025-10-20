use std::fmt::Debug;

pub type DefaultValue = i32;

pub trait Value: Sized + Send + Sync + Debug + Clone + PartialEq + Eq + PartialOrd + Ord {}

pub trait OrderedValue: PartialOrd + Ord + Value {}

pub trait ComputableValue: Value {
    fn compute(&self) -> f64;
}

macro_rules! impl_computable_value_from {
    ($typ:ty) => {
        impl $crate::value::ComputableValue for $typ {
            fn compute(&self) -> f64 {
                f64::from(*self)
            }
        }
    };
}

macro_rules! impl_computable_value_trunc {
    ($typ:ty) => {
        impl $crate::value::ComputableValue for $typ {
            fn compute(&self) -> f64 {
                *self as f64
            }
        }
    };
}

impl<T> Value for T where T: Sized + Send + Sync + Debug + Clone + PartialEq + Eq + PartialEq + Ord {}

impl<T> OrderedValue for T where T: Value + PartialOrd + Ord {}

impl_computable_value_from!(u8);
impl_computable_value_from!(u16);
impl_computable_value_from!(u32);
impl_computable_value_trunc!(u64);
impl_computable_value_trunc!(u128);
impl_computable_value_from!(i8);
impl_computable_value_from!(i16);
impl_computable_value_from!(i32);
impl_computable_value_trunc!(i64);
impl_computable_value_trunc!(i128);
