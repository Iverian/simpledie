use std::sync::LazyLock;

use bon::Builder;
use modron::{DefaultValue, Die};

use crate::{d20, D20};

pub static ADV: LazyLock<Die> = LazyLock::new(|| D20.nmax(2));
pub static DIS: LazyLock<Die> = LazyLock::new(|| D20.nmin(2));

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Attack {
    Critical,
    Hit,
    Miss,
}

#[derive(Builder, Debug, Clone)]
pub struct AttackRoll {
    #[builder(default = 10)]
    armor_class: DefaultValue,
    #[builder(default = d20())]
    die: Die,
    atk_die: Option<Die>,
    #[builder(default = 0)]
    ability_bonus: DefaultValue,
    #[builder(default = 0)]
    proficiency: DefaultValue,
    #[builder(default = 0)]
    atk_bonus: DefaultValue,
    #[builder(default = 20)]
    critical_on: DefaultValue,
}

#[derive(Builder, Debug, Clone)]
pub struct WeaponDamageRoll {
    #[builder(start_fn)]
    dmg_die: Die,
    #[builder(default = 10)]
    armor_class: DefaultValue,
    #[builder(default = d20())]
    die: Die,
    atk_die: Option<Die>,
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

impl AttackRoll {
    pub fn eval(self) -> Die<Attack> {
        let atk_bonus = self.ability_bonus + self.proficiency + self.atk_bonus;
        self.die.apply_two(
            &self.atk_die.unwrap_or_else(|| Die::scalar(0)),
            |&atk, &atk_extra| match atk {
                x if x >= self.critical_on => Attack::Critical,
                x if x + atk_extra + atk_bonus >= self.armor_class => Attack::Hit,
                _ => Attack::Miss,
            },
        )
    }
}

impl WeaponDamageRoll {
    pub fn eval(self) -> Die {
        let dmg_bonus = self.ability_bonus + self.dmg_bonus + self.weapon_bonus;
        AttackRoll {
            armor_class: self.armor_class,
            die: self.die,
            atk_die: self.atk_die,
            ability_bonus: self.ability_bonus,
            proficiency: self.proficiency,
            atk_bonus: self.atk_bonus + self.weapon_bonus,
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

pub fn adv() -> Die {
    ADV.clone()
}

pub fn dis() -> Die {
    DIS.clone()
}
