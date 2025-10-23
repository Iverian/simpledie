use std::sync::LazyLock;

use bon::Builder;
use modron::{DefaultValue, Die};

use crate::{d20, zero, D20};

pub static ADV: LazyLock<Die> = LazyLock::new(|| D20.nmax(2));
pub static DIS: LazyLock<Die> = LazyLock::new(|| D20.nmin(2));

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Attack {
    Critical,
    Hit,
    Miss,
}

#[derive(Builder, Debug, Clone)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct AttackRoll {
    #[builder(start_fn)]
    armor_class: DefaultValue,
    #[builder(default = d20())]
    die: Die,
    extra_die: Option<Die>,
    #[builder(default = 0)]
    ability_bonus: DefaultValue,
    #[builder(default = 0)]
    proficiency: DefaultValue,
    #[builder(default = 0)]
    bonus: DefaultValue,
    #[builder(default = 20)]
    critical_on: DefaultValue,
}

#[derive(Builder, Debug, Clone)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct WeaponDamageRoll {
    #[builder(start_fn)]
    armor_class: DefaultValue,
    #[builder(start_fn)]
    dmg_die: Die,
    #[builder(default = d20())]
    die: Die,
    extra_atk_die: Option<Die>,
    #[builder(default = 0)]
    ability_bonus: DefaultValue,
    #[builder(default = 0)]
    proficiency: DefaultValue,
    #[builder(default = 0)]
    atk_bonus: DefaultValue,
    #[builder(default = 0)]
    dmg_bonus: DefaultValue,
    #[builder(default = 0)]
    weapon_bonus: DefaultValue,
    #[builder(default = 20)]
    critical_on: DefaultValue,
}

#[derive(Builder, Debug, Clone)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct SavingThrow {
    #[builder(start_fn)]
    difficulty: DefaultValue,
    #[builder(default = d20())]
    die: Die,
    extra_die: Option<Die>,
    #[builder(default = 0)]
    bonus: DefaultValue,
}

#[derive(Builder, Debug, Clone)]
#[builder(start_fn(name = "new"), finish_fn(vis = ""))]
pub struct SavingThrowDamage {
    #[builder(start_fn)]
    difficulty: DefaultValue,
    #[builder(start_fn)]
    dmg_die: Die,
    half_dmg: bool,
    #[builder(default = 1)]
    targets: usize,
    #[builder(default = d20())]
    die: Die,
    extra_die: Option<Die>,
    save_bonus: DefaultValue,
}

impl AttackRoll {
    pub fn eval(self) -> Die<Attack> {
        let atk_bonus = self.ability_bonus + self.proficiency + self.bonus;
        self.die.apply_two(
            &self.extra_die.unwrap_or_else(zero),
            |&atk, &atk_extra| match atk {
                x if x >= self.critical_on => Attack::Critical,
                x if x + atk_extra + atk_bonus >= self.armor_class => Attack::Hit,
                _ => Attack::Miss,
            },
        )
    }
}

impl<S> AttackRollBuilder<S>
where
    S: attack_roll_builder::State,
    S: attack_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die<Attack> {
        self.build().eval()
    }
}

impl WeaponDamageRoll {
    pub fn eval(self) -> Die {
        let dmg_bonus = self.ability_bonus + self.dmg_bonus + self.weapon_bonus;
        AttackRoll {
            armor_class: self.armor_class,
            die: self.die,
            extra_die: self.extra_atk_die,
            ability_bonus: self.ability_bonus,
            proficiency: self.proficiency,
            bonus: self.atk_bonus + self.weapon_bonus,
            critical_on: self.critical_on,
        }
        .eval()
        .apply_three(
            &self.dmg_die,
            &self.dmg_die,
            |&atk, &dmg, &dmg_crit| match atk {
                Attack::Critical => dmg + dmg_crit + dmg_bonus,
                Attack::Hit => dmg + dmg_bonus,
                Attack::Miss => 0,
            },
        )
    }
}

impl<S> WeaponDamageRollBuilder<S>
where
    S: weapon_damage_roll_builder::State,
    S: weapon_damage_roll_builder::IsComplete,
{
    pub fn eval(self) -> Die {
        self.build().eval()
    }
}

impl SavingThrow {
    pub fn eval(self) -> Die<bool> {
        self.die
            .apply_two(&self.extra_die.unwrap_or_else(zero), |&die, &extra| {
                die + extra + self.bonus >= self.difficulty
            })
    }
}

impl<S> SavingThrowBuilder<S>
where
    S: saving_throw_builder::State,
    S: saving_throw_builder::IsComplete,
{
    pub fn eval(self) -> Die<bool> {
        self.build().eval()
    }
}

impl SavingThrowDamage {
    pub fn eval(self) -> Die {
        SavingThrow {
            difficulty: self.difficulty,
            die: self.die,
            extra_die: self.extra_die,
            bonus: self.save_bonus,
        }
        .eval()
        .apply_two(&self.dmg_die, |&save, &dmg| {
            if save {
                if self.half_dmg {
                    dmg / 2
                } else {
                    0
                }
            } else {
                dmg
            }
        })
        .nsum(self.targets.max(1))
    }
}

impl<S> SavingThrowDamageBuilder<S>
where
    S: saving_throw_damage_builder::State,
    S: saving_throw_damage_builder::IsComplete,
{
    pub fn eval(self) -> Die {
        self.build().eval()
    }
}

pub fn adv() -> Die {
    ADV.clone()
}

pub fn dis() -> Die {
    DIS.clone()
}
