use once_cell::sync::Lazy;

use crate::Die;

pub static D0: Lazy<Die<u32>> = Lazy::new(|| Die::single(0));
pub static D2: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(2));
pub static D3: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(3));
pub static D4: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(4));
pub static D6: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(6));
pub static D8: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(8));
pub static D10: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(10));
pub static D12: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(12));
pub static D20: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(20));
pub static D100: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(100));

pub fn d0() -> Die<u32> {
    D0.clone()
}

pub fn d2() -> Die<u32> {
    D2.clone()
}

pub fn d3() -> Die<u32> {
    D3.clone()
}

pub fn d4() -> Die<u32> {
    D4.clone()
}

pub fn d6() -> Die<u32> {
    D6.clone()
}

pub fn d8() -> Die<u32> {
    D8.clone()
}

pub fn d10() -> Die<u32> {
    D10.clone()
}

pub fn d12() -> Die<u32> {
    D12.clone()
}

pub fn d20() -> Die<u32> {
    D20.clone()
}

pub fn d100() -> Die<u32> {
    D100.clone()
}
