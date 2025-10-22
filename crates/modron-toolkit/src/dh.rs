use std::fmt::Display;
use std::ops::{Add, Sub};
use std::sync::LazyLock;

use bon::Builder;
use modron::{ComputableValue, DefaultValue, Die};

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
pub enum ActionRollResult {
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
    fn compute(&self) -> i128 {
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

impl ComputableValue for ActionRollResult {
    fn compute(&self) -> i128 {
        match self {
            ActionRollResult::FailureWithFear => -2,
            ActionRollResult::SuccessWithFear => 1,
            ActionRollResult::FailureWithHope => -1,
            ActionRollResult::SuccessWithHope => 2,
            ActionRollResult::Critical => 3,
        }
    }
}

impl Display for ActionRollResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionRollResult::FailureWithFear => write!(f, "FF"),
            ActionRollResult::SuccessWithFear => write!(f, "SF"),
            ActionRollResult::FailureWithHope => write!(f, "FH"),
            ActionRollResult::SuccessWithHope => write!(f, "SH"),
            ActionRollResult::Critical => write!(f, "C"),
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
#[builder(finish_fn(name = "options", vis = ""))]
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
#[builder(finish_fn(name = "options", vis = ""))]
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
#[builder(finish_fn(name = "options", vis = ""))]
pub struct AttackRoll {
    #[builder(start_fn)]
    pub dmg_die: Die,
    #[builder(default = 10)]
    pub difficulty: i32,
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
    pub fn eval(self) -> Die<ActionRollResult> {
        match (self.help, self.modifier) {
            (true, Some(ExtMod::Adv)) => self.die + d6().nmax(2),
            (true, None) | (false, Some(ExtMod::Adv)) => self.die + d6(),
            (true, Some(ExtMod::Dis)) | (false, None) => self.die,
            (false, Some(ExtMod::Dis)) => self.die - d6(),
        }
        .map(|&a| match a {
            Duality::Fear(x) => {
                if x + self.bonus >= self.difficulty {
                    ActionRollResult::SuccessWithFear
                } else {
                    ActionRollResult::FailureWithFear
                }
            }
            Duality::Hope(x) => {
                if x + self.bonus >= self.difficulty {
                    ActionRollResult::SuccessWithHope
                } else {
                    ActionRollResult::SuccessWithFear
                }
            }
            Duality::Critical => ActionRollResult::Critical,
        })
    }
}

impl<S> ActionRollBuilder<S>
where
    S: action_roll_builder::State,
    S: action_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die<ActionRollResult> {
        self.options().eval()
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

impl AttackRoll {
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
            ActionRollResult::FailureWithHope | ActionRollResult::FailureWithFear => 0,
            ActionRollResult::SuccessWithHope | ActionRollResult::SuccessWithFear => {
                dmg + self.dmg_bonus
            }
            ActionRollResult::Critical => dmg + self.dmg_bonus + crit_bonus,
        })
    }
}

impl<S> AttackRollBuilder<S>
where
    S: attack_roll_builder::State,
    S: attack_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die {
        self.options().eval()
    }
}
