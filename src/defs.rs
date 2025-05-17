use std::sync::LazyLock;

use crate::die::Die;

pub static D1: LazyLock<Die<i32>> = LazyLock::new(|| Die::single(1));
pub static D2: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(2));
pub static D3: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(3));
pub static D4: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(4));
pub static D6: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(6));
pub static D8: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(8));
pub static D10: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(10));
pub static D12: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(12));
pub static D20: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(20));
pub static D100: LazyLock<Die<i32>> = LazyLock::new(|| Die::uniform(100));

pub fn d1() -> Die<i32> {
    D1.clone()
}

pub fn d2() -> Die<i32> {
    D2.clone()
}

pub fn d3() -> Die<i32> {
    D3.clone()
}

pub fn d4() -> Die<i32> {
    D4.clone()
}

pub fn d6() -> Die<i32> {
    D6.clone()
}

pub fn d8() -> Die<i32> {
    D8.clone()
}

pub fn d10() -> Die<i32> {
    D10.clone()
}

pub fn d12() -> Die<i32> {
    D12.clone()
}

pub fn d20() -> Die<i32> {
    D20.clone()
}

pub fn d100() -> Die<i32> {
    D100.clone()
}
