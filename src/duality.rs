use crate::{ComputableValue, value::DefaultValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DualityResult {
    Fear(DefaultValue),
    Hope(DefaultValue),
    Critical,
}

impl DualityResult {
    #[must_use]
    pub fn new(hope: DefaultValue, fear: DefaultValue) -> Self {
        if hope == fear {
            DualityResult::Critical
        } else if hope > fear {
            DualityResult::Hope(hope + fear)
        } else {
            DualityResult::Fear(hope + fear)
        }
    }
}

impl ComputableValue for DualityResult {
    fn compute(&self) -> f64 {
        match self {
            DualityResult::Fear(x) => -x.compute(),
            DualityResult::Hope(x) => x.compute(),
            DualityResult::Critical => 0f64,
        }
    }
}

impl<T> std::ops::Add<T> for DualityResult
where
    DefaultValue: std::ops::Add<T, Output = DefaultValue>,
{
    type Output = DualityResult;

    fn add(self, rhs: T) -> Self::Output {
        match self {
            DualityResult::Fear(x) => DualityResult::Fear(x.add(rhs)),
            DualityResult::Hope(x) => DualityResult::Hope(x.add(rhs)),
            DualityResult::Critical => DualityResult::Critical,
        }
    }
}

impl<T> std::ops::Sub<T> for DualityResult
where
    DefaultValue: std::ops::Sub<T, Output = DefaultValue>,
{
    type Output = DualityResult;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
            DualityResult::Fear(x) => DualityResult::Fear(x.sub(rhs)),
            DualityResult::Hope(x) => DualityResult::Hope(x.sub(rhs)),
            DualityResult::Critical => DualityResult::Critical,
        }
    }
}
