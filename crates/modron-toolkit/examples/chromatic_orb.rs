use std::iter::repeat_n;

use modron_toolkit::*;

fn main() {
    let level = 5;
    let targets = 6;
    let atk_die = dnd::AttackRoll::new(15)
        .ability_bonus(5)
        .proficiency(4)
        .eval();

    let values = chromatic_orb_mean(&atk_die, level, targets);
    println!("Chromatic orb Lv.{level} to {targets} targets");
    for (target, (mean, stddev)) in values.into_iter().enumerate() {
        println!("{:3} | {:.2} ± {:.2}", target + 1, mean, stddev);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
struct OrbRaw {
    dmg: DefaultValue,
    bounce: bool,
    values: u8,
}

#[derive(Clone, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
struct Orb {
    dmg: Vec<DefaultValue>,
    bounce: bool,
}

impl OrbRaw {
    fn new(dice: &[&DefaultValue]) -> Self {
        let mut dmg = 0;
        let mut values = 0;
        let mut bits = 0;
        let mut bounce = false;
        for &&i in dice {
            let bit = 1u8 << (i as u8 - 1);
            values |= bit;
            bits ^= bit;
            bounce = bounce || bits & bit == 0;
            dmg += i;
        }

        Self {
            dmg,
            bounce,
            values,
        }
    }

    fn base(&self) -> Orb {
        Orb {
            dmg: vec![self.dmg],
            bounce: self.bounce,
        }
    }

    fn critical(&self, other: &Self) -> Orb {
        Orb {
            dmg: vec![self.dmg + other.dmg],
            bounce: self.bounce || other.bounce || bitmatch(self.values, other.values),
        }
    }
}

impl Orb {
    fn append(&self, other: &Self) -> Self {
        let mut dmg = self.dmg.clone();
        dmg.extend(other.dmg.iter().copied());
        Orb {
            dmg,
            bounce: other.bounce,
        }
    }
}

#[inline]
fn bitmatch(lhs: u8, rhs: u8) -> bool {
    for i in 0..8u8 {
        let b = 1u8 << i;
        if (lhs & b != 0) && (rhs & b != 0) {
            return true;
        }
    }
    false
}

fn chromatic_orb_dmg(level: usize) -> Die<OrbRaw> {
    let n = 2 + level;
    Die::apply(vec![d8(); n], OrbRaw::new)
}

fn chromatic_orb_one(atk_die: &Die<dnd::Attack>, level: usize) -> Die<Orb> {
    let dmg_die = chromatic_orb_dmg(level);
    atk_die.apply_three(&dmg_die, &dmg_die, |&atk, dmg, crit| match atk {
        dnd::Attack::Critical => dmg.critical(crit),
        dnd::Attack::Hit => dmg.base(),
        dnd::Attack::Miss => Orb::default(),
    })
}

fn chromatic_orb(atk_die: &Die<dnd::Attack>, level: usize, targets: usize) -> Die<Vec<i32>> {
    let targets = targets.clamp(1, 1 + level);
    chromatic_orb_one(atk_die, level)
        .explode_one(targets, |x| x.last().unwrap().bounce, |x, y| x.append(y))
        .map(|x| {
            x.dmg
                .iter()
                .copied()
                .chain(repeat_n(0, targets - x.dmg.len()))
                .collect()
        })
}

fn chromatic_orb_mean(atk_die: &Die<dnd::Attack>, level: usize, targets: usize) -> Vec<(f64, f64)> {
    let targets = targets.clamp(1, 1 + level);
    let die = chromatic_orb(atk_die, level, targets);
    let mut result = Vec::with_capacity(targets);
    for t in 0..targets {
        let d = die.map(|x| x[t]);
        result.push((d.mean(), d.stddev()));
    }
    result
}
