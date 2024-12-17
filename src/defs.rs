use once_cell::sync::Lazy;

use crate::Die;

pub static D2: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(2));
pub static D3: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(3));
pub static D4: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(4));
pub static D6: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(6));
pub static D8: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(8));
pub static D12: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(12));
pub static D20: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(20));
pub static D100: Lazy<Die<u32>> = Lazy::new(|| Die::uniform(100));
