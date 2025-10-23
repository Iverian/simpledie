use std::sync::LazyLock;

use modron::Die;

pub static ZERO: LazyLock<Die> = LazyLock::new(|| Die::scalar(0));
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

pub fn zero() -> Die {
    ZERO.clone()
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

pub fn d100() -> Die {
    D100.clone()
}
