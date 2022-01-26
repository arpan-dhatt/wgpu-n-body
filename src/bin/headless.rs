use std::time::Instant;

use wgpu_n_body::{runners::OfflineHeadless, inits, sims::{NaiveSim, SimParams, TreeSim}};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

const STEPS: usize = 10;

fn main() {
    let sim_params = SimParams {
        particle_num: 1000000,
        g: 0.000001,
        e: 0.0001,
        dt: 0.016,
    };
    println!("Initializing Simulation");
    let mut runner = pollster::block_on(OfflineHeadless::<TreeSim>::new(sim_params, inits::disc_init)).unwrap();
    println!("Running Simulation");
    for _ in 0..STEPS {
        let now = Instant::now();
        runner.step();
        println!("Step Duration: {} µs", now.elapsed().as_micros());
    }
    println!("Finished Running");
}
