use std::num::NonZeroU16;
use std::sync::LazyLock;

use crate::Die;

pub static D1: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(1).unwrap()));
pub static D2: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(2).unwrap()));
pub static D3: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(3).unwrap()));
pub static D4: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(4).unwrap()));
pub static D6: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(6).unwrap()));
pub static D8: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(8).unwrap()));
pub static D10: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(10).unwrap()));
pub static D12: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(12).unwrap()));
pub static D20: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(20).unwrap()));
pub static D100: LazyLock<Die> = LazyLock::new(|| Die::uniform(NonZeroU16::new(100).unwrap()));

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
