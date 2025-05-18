use std::time::Instant;

use simpledie::prelude::*;

fn main() {
    let start = Instant::now();
    let d = d10().sum(9).le(50).try_eval().unwrap();
    let elapsed = (Instant::now() - start).as_millis();
    let mean = d.mean_f64();
    println!("die = {d:?} mean = {mean} elapsed = {elapsed}");
}
