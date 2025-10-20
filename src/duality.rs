use std::ops::{Add, Sub};

use crate::value::DefaultValue;
use crate::ComputableValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Duality {
    Fear(DefaultValue),
    Hope(DefaultValue),
    Critical,
}

impl Duality {
    #[must_use]
    pub fn new(hope: DefaultValue, fear: DefaultValue) -> Self {
        if hope == fear {
            Duality::Critical
        } else if hope > fear {
            Duality::Hope(hope + fear)
        } else {
            Duality::Fear(hope + fear)
        }
    }
}

impl ComputableValue for Duality {
    fn compute(&self) -> f64 {
        match self {
            Duality::Fear(x) => -x.compute(),
            Duality::Hope(x) => x.compute(),
            Duality::Critical => 0f64,
        }
    }
}

impl<T> Add<T> for Duality
where
    DefaultValue: Add<T, Output = DefaultValue>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        match self {
            Duality::Fear(x) => Duality::Fear(x.add(rhs)),
            Duality::Hope(x) => Duality::Hope(x.add(rhs)),
            Duality::Critical => Duality::Critical,
        }
    }
}

impl<T> Sub<T> for Duality
where
    DefaultValue: Sub<T, Output = DefaultValue>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
            Duality::Fear(x) => Duality::Fear(x.sub(rhs)),
            Duality::Hope(x) => Duality::Hope(x.sub(rhs)),
            Duality::Critical => Duality::Critical,
        }
    }
}
