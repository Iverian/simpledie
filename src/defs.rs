use std::sync::LazyLock;

use crate::die::Die;
use crate::duality::DualityResult;

pub static D0: LazyLock<Die> = LazyLock::new(Die::zero);
pub static D1: LazyLock<Die> = LazyLock::new(|| Die::scalar(1));
pub static D2: LazyLock<Die> = LazyLock::new(|| Die::numeric(2));
pub static D3: LazyLock<Die> = LazyLock::new(|| Die::numeric(3));
pub static D4: LazyLock<Die> = LazyLock::new(|| Die::numeric(4));
pub static D6: LazyLock<Die> = LazyLock::new(|| Die::numeric(6));
pub static D8: LazyLock<Die> = LazyLock::new(|| Die::numeric(8));
pub static D10: LazyLock<Die> = LazyLock::new(|| Die::numeric(10));
pub static D12: LazyLock<Die> = LazyLock::new(|| Die::numeric(12));
pub static D20: LazyLock<Die> = LazyLock::new(|| Die::numeric(20));
pub static D100: LazyLock<Die> = LazyLock::new(|| Die::numeric(100));
pub static D20KH: LazyLock<Die> = LazyLock::new(|| D20.apply_two(&D20, |&x, &y| x.max(y)));
pub static D20KL: LazyLock<Die> = LazyLock::new(|| D20.apply_two(&D20, |&x, &y| x.min(y)));
pub static DUAL: LazyLock<Die<DualityResult>> =
    LazyLock::new(|| D12.apply_two(&D12, |&h, &f| DualityResult::new(h, f)));

pub fn d0() -> Die {
    D0.clone()
}

pub fn d1() -> Die {
    D1.clone()
}

pub fn d2() -> Die {
    D2.clone()
}

pub fn d3() -> Die {
    D3.clone()
}

pub fn d4() -> Die {
    D4.clone()
}

pub fn d6() -> Die {
    D6.clone()
}

pub fn d8() -> Die {
    D8.clone()
}

pub fn d10() -> Die {
    D10.clone()
}

pub fn d12() -> Die {
    D12.clone()
}

pub fn d20() -> Die {
    D20.clone()
}

pub fn d20kh() -> Die {
    D20KH.clone()
}

pub fn d20kl() -> Die {
    D20KL.clone()
}

pub fn dual() -> Die<DualityResult> {
    DUAL.clone()
}
