use std::time::Instant;

use wgpu_n_body::{
    inits,
    runners::OfflineHeadless,
    sims::{SimParams, TreeSim, AddParams},
};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

const STEPS: usize = 10;

fn main() {
    let sim_params = SimParams {
        particle_num: 40,
        g: 0.000001,
        e: 0.0001,
        dt: 0.016,
    };
    println!("Initializing Simulation");
    let mut runner = pollster::block_on(OfflineHeadless::<TreeSim>::new(
        sim_params,
        AddParams::TreeSimParams { theta: 0.75 },
        inits::uniform_init,
    ))
    .unwrap();
    println!("Running Simulation");
    for _ in 0..STEPS {
        let now = Instant::now();
        runner.step();
        println!("Step Duration: {} µs", now.elapsed().as_micros());
    }
    println!("Finished Running");
}
