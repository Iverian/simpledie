use simpledie::prelude::*;

fn main() {
    combo_strikes();
}

fn duality_attack() {
    let difficulty = 15;
    let proficiency = 2;
    let dmg_die = proficiency * (d10() + d6());
    let dmg_bonus = 3;

    let dmg = dual().apply_two(&dmg_die, |&a, &d| match a {
        Duality::Fear(x) | Duality::Hope(x) if x >= difficulty => d + dmg_bonus,
        Duality::Critical => dmg_die.max_value() + d + dmg_bonus,
        _ => 0,
    });
    println!("{dmg:?}, mean = {}", dmg.mean())
}

fn combo_strikes() {
    let combo_die = d2();
    for m in [2, 5, 10] {
        let cd = combo_die.explode(
            1..m,
            |x| {
                let n = x.len();
                if n <= 1 {
                    true
                } else {
                    x[n - 1] >= x[n - 2]
                }
            },
            |&x, &y| x + y,
        );

        println!(
            "m = {} dice = {:?} mean = {} f = {}",
            m,
            cd,
            cd.mean(),
            cd.ge(*combo_die.max_value()).mean()
        );
    }
}
