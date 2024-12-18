use once_cell::sync::Lazy;

use crate::Die;

pub static D0: Lazy<Die<i32>> = Lazy::new(|| Die::single(0));
pub static D2: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(2));
pub static D3: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(3));
pub static D4: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(4));
pub static D6: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(6));
pub static D8: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(8));
pub static D10: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(10));
pub static D12: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(12));
pub static D20: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(20));
pub static D100: Lazy<Die<i32>> = Lazy::new(|| Die::uniform(100));

pub fn d0() -> Die<i32> {
    D0.clone()
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
