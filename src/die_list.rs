use std::fmt::Debug;
use std::iter::{Product, Sum};

use super::die::Die;

#[derive(Debug)]
pub struct DieList<K> {
    value: Vec<Die<K>>,
}

impl<K> DieList<K> {
    #[must_use]
    pub fn empty() -> Self {
        Self { value: Vec::new() }
    }

    pub fn push<U>(&mut self, die: Die<U>)
    where
        K: From<U>,
    {
        self.value.push(die.cast());
    }

    pub fn extend<U>(&mut self, die: DieList<U>)
    where
        K: From<U>,
    {
        self.value.extend(die.value.into_iter().map(Die::cast));
    }

    #[must_use]
    pub fn combine<F, U>(&self, op: F) -> Die<U>
    where
        F: Fn(&[&K]) -> U,
        U: Ord,
    {
        Die::combine(self.value.as_slice(), op)
    }

    #[must_use]
    pub fn map<F, U>(self, op: F) -> DieList<U>
    where
        F: Fn(K) -> U + Copy,
        U: Ord,
    {
        DieList {
            value: self.value.into_iter().map(|x| x.map(op)).collect(),
        }
    }

    #[must_use]
    pub fn cast<T>(self) -> DieList<T>
    where
        T: From<K>,
    {
        DieList {
            value: self.value.into_iter().map(Die::cast).collect(),
        }
    }

    pub fn try_cast<T>(self) -> core::result::Result<DieList<T>, <T as TryFrom<K>>::Error>
    where
        T: TryFrom<K>,
        <T as TryFrom<K>>::Error: Clone + Debug,
    {
        self.value
            .into_iter()
            .map(Die::try_cast)
            .collect::<core::result::Result<Vec<_>, _>>()
            .map(|value| DieList { value })
    }
}

impl<K> DieList<K>
where
    K: Clone,
{
    #[must_use]
    pub fn repeat(count: usize, die: Die<K>) -> Self {
        Self {
            value: vec![die; count],
        }
    }
}

impl<K> DieList<K>
where
    K: Ord + Clone,
{
    #[must_use]
    pub fn reduce<F>(&self, op: F) -> Option<Die<K>>
    where
        F: Fn(&K, &K) -> K + Copy,
    {
        if self.value.is_empty() {
            return None;
        }
        let mut result = self.value[0].clone();
        for x in &self.value[1..] {
            result = result.combine_with(x, |x, y| op(x, y).clone());
        }
        Some(result)
    }
}

impl<K> DieList<K>
where
    K: Ord + Sum + Clone,
{
    #[must_use]
    pub fn sum(&self) -> Die<K> {
        self.combine(|x| x.iter().map(|&x| x.clone()).sum())
    }
}

impl<K> DieList<K>
where
    K: Ord + Product + Clone,
{
    #[must_use]
    pub fn product(&self) -> Die<K> {
        self.combine(|x| x.iter().map(|&x| x.clone()).product())
    }
}

impl<K> DieList<K>
where
    K: Ord + Clone + Default,
{
    #[must_use]
    pub fn min(&self) -> Die<K> {
        self.combine(|x| x.iter().map(|&x| x.clone()).min().unwrap_or_default())
    }

    #[must_use]
    pub fn max(&self) -> Die<K> {
        self.combine(|x| x.iter().map(|&x| x.clone()).max().unwrap_or_default())
    }
}

impl<K> DieList<K>
where
    K: Clone + Into<bool>,
{
    #[must_use]
    pub fn bool_any(&self) -> Die<bool> {
        self.combine(|x| x.iter().any(|&x| x.clone().into()))
    }

    #[must_use]
    pub fn bool_all(&self) -> Die<bool> {
        self.combine(|x| x.iter().all(|&x| x.clone().into()))
    }

    #[must_use]
    pub fn bool_count(&self) -> Die<usize> {
        self.combine(|x| x.iter().filter(|&&x| x.clone().into()).count())
    }
}
