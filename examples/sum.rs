use std::f64;
use std::time::Instant;

use simpledie::prelude::*;

fn main() {
    let start = Instant::now();
    let d = d10().sum(10).lt(50).eval();
    let elapsed = (Instant::now() - start).as_secs_f64();
    let mean = d.mean().unwrap_or(f64::NAN);
    println!("dir = {d:?} mean = {mean} elapsed = {elapsed}");
}
