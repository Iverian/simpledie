use std::f64;
use std::time::Instant;

use simpledie::prelude::*;

fn main() {
    #[cfg(feature = "parallel")]
    {
        println!("threads = {}", rayon::current_num_threads());
    }

    let start = Instant::now();
    let d = d10().sum(9).le(50).try_eval().unwrap();
    let elapsed = (Instant::now() - start).as_millis();
    let mean = d.mean().unwrap_or(f64::NAN);
    println!("die = {d:?} mean = {mean} elapsed = {elapsed}");
}
