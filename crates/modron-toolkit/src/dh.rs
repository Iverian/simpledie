use std::fmt::Display;
use std::ops::{Add, Sub};
use std::sync::LazyLock;

use bon::Builder;
use modron::{ComputableValue, ComputedValue, DefaultValue, Die};

use crate::{d6, D12};

pub static DUAL: LazyLock<Die<Duality>> =
    LazyLock::new(|| D12.apply_two(&D12, |&h, &f| Duality::new(h, f)));

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Duality {
    Fear(DefaultValue),
    Hope(DefaultValue),
    Critical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    FailureWithFear,
    SuccessWithFear,
    FailureWithHope,
    SuccessWithHope,
    Critical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExtMod {
    Adv,
    Dis,
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
    fn compute(&self) -> ComputedValue {
        match self {
            Duality::Fear(x) => -x.compute(),
            Duality::Hope(x) => x.compute(),
            Duality::Critical => 25,
        }
    }
}

impl Display for Duality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Duality::Fear(x) => write!(f, "F{x}"),
            Duality::Hope(x) => write!(f, "H{x}"),
            Duality::Critical => write!(f, "C"),
        }
    }
}

impl ComputableValue for Action {
    fn compute(&self) -> ComputedValue {
        match self {
            Action::FailureWithFear => -2,
            Action::SuccessWithFear => 1,
            Action::FailureWithHope => -1,
            Action::SuccessWithHope => 2,
            Action::Critical => 3,
        }
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::FailureWithFear => write!(f, "FF"),
            Action::SuccessWithFear => write!(f, "SF"),
            Action::FailureWithHope => write!(f, "FH"),
            Action::SuccessWithHope => write!(f, "SH"),
            Action::Critical => write!(f, "C"),
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

pub fn dual() -> Die<Duality> {
    DUAL.clone()
}

#[derive(Builder, Clone, Debug)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct ActionRoll {
    #[builder(default = 10)]
    pub difficulty: i32,
    #[builder(default = dual())]
    pub die: Die<Duality>,
    #[builder(default = 0)]
    pub bonus: DefaultValue,
    #[builder(default = false)]
    pub help: bool,
    pub modifier: Option<ExtMod>,
}

#[derive(Builder, Clone, Debug)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct ReactionRoll {
    #[builder(default = 10)]
    pub difficulty: i32,
    #[builder(default = dual())]
    pub die: Die<Duality>,
    #[builder(default = 0)]
    pub bonus: DefaultValue,
    pub modifier: Option<ExtMod>,
}

#[derive(Builder, Clone, Debug)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct AttackDamageRoll {
    #[builder(start_fn)]
    pub dmg_die: Die,
    #[builder(default = 10)]
    pub difficulty: DefaultValue,
    #[builder(default = dual())]
    pub die: Die<Duality>,
    #[builder(default = 0)]
    pub dmg_bonus: DefaultValue,
    #[builder(default = 1)]
    pub proficiency: usize,
    #[builder(default = 0)]
    pub bonus: DefaultValue,
    #[builder(default = false)]
    pub help: bool,
    pub modifier: Option<ExtMod>,
}

impl ActionRoll {
    pub fn eval(self) -> Die<Action> {
        match (self.help, self.modifier) {
            (true, Some(ExtMod::Adv)) => self.die + d6().nmax(2),
            (true, None) | (false, Some(ExtMod::Adv)) => self.die + d6(),
            (true, Some(ExtMod::Dis)) | (false, None) => self.die,
            (false, Some(ExtMod::Dis)) => self.die - d6(),
        }
        .map(|&a| match a {
            Duality::Fear(x) => {
                if x + self.bonus >= self.difficulty {
                    Action::SuccessWithFear
                } else {
                    Action::FailureWithFear
                }
            }
            Duality::Hope(x) => {
                if x + self.bonus >= self.difficulty {
                    Action::SuccessWithHope
                } else {
                    Action::SuccessWithFear
                }
            }
            Duality::Critical => Action::Critical,
        })
    }
}

impl<S> ActionRollBuilder<S>
where
    S: action_roll_builder::State,
    S: action_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die<Action> {
        self.build().eval()
    }
}

impl ReactionRoll {
    pub fn eval(self) -> Die<bool> {
        match self.modifier {
            Some(ExtMod::Adv) => self.die + d6(),
            Some(ExtMod::Dis) => self.die - d6(),
            None => dual(),
        }
        .map(|&a| match a {
            Duality::Hope(x) | Duality::Fear(x) if x + self.bonus >= self.difficulty => true,
            Duality::Critical => true,
            _ => false,
        })
    }
}

impl AttackDamageRoll {
    pub fn eval(self) -> Die {
        let dmg_die = self.proficiency * self.dmg_die;
        let crit_bonus = dmg_die.max_value();

        ActionRoll {
            die: self.die,
            difficulty: self.difficulty,
            bonus: self.bonus,
            help: self.help,
            modifier: self.modifier,
        }
        .eval()
        .apply_two(&dmg_die, |&atk, &dmg| match atk {
            Action::FailureWithHope | Action::FailureWithFear => 0,
            Action::SuccessWithHope | Action::SuccessWithFear => dmg + self.dmg_bonus,
            Action::Critical => dmg + self.dmg_bonus + crit_bonus,
        })
    }
}

impl<S> AttackDamageRollBuilder<S>
where
    S: attack_damage_roll_builder::State,
    S: attack_damage_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die {
        self.build().eval()
    }
}
