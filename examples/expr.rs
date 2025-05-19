use std::iter::{once, repeat_n};

use simpledie::prelude::*;

fn main() {
    todo!()
}

fn atk(die: impl Expr, atk_bonus: i32, armor_class: i32) -> impl Expr {
    die.map(move |x| match x {
        20 => 2,
        x if x != 1 && x + atk_bonus >= armor_class => 1,
        _ => 0,
    })
}

fn atk_dmg<A, D>(atk: A, dmg: D, dmg_bonus: i32, multiattack: usize) -> impl Expr
where
    A: Expr,
    D: Expr + Clone,
    A::Op: Clone,
    D::Op: Clone,
{
    atk.fold_three(dmg.clone(), dmg, move |x, y, z| match x {
        2 => y + z + dmg_bonus,
        1 => y + dmg_bonus,
        _ => 0,
    })
    .sum_n(multiattack)
}

fn save(die: impl Expr, save_bonus: i32, save_dc: i32) -> impl Expr {
    die.kadd(save_bonus).gt(save_dc)
}

fn save_dmg(save: impl Expr, dmg: impl Expr, half: bool) -> impl Expr {
    save.fold_two(dmg, move |x, y| {
        if x != 0 {
            y
        } else if half {
            y / 2
        } else {
            0
        }
    })
}

fn fireball(save: impl Expr) -> impl Expr {
    save_dmg(save, d6().sum_n(8), true)
}

fn sorc_burst(
    die: impl Expr,
    cha: i32,
    init: usize,
    pb: i32,
    weapon_plus: i32,
    armor_class: i32,
) -> impl Expr {
    Die::fold(
        once(atk(die, cha + pb + weapon_plus, armor_class).eval())
            .chain(repeat_n(d8(), 2 * init + cha as usize)),
        move |x| match x[0] {
            2 => sorc_burst_hit(&x[1..], init),
            1 => sorc_burst_hit(&x[1..(1 + init + cha as usize)], init),
            _ => 0,
        },
    )
}

fn sorc_burst_hit(x: &[i32], mut dice: usize) -> i32 {
    let mut result = 0;
    for (i, &d) in x.iter().enumerate() {
        if i == dice {
            break;
        }
        result += d;
        if d == 8 {
            dice += 1;
        }
    }
    result
}
