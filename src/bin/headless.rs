use wgpu_n_body::{runners::OfflineHeadless, inits, sims::{NaiveSim, SimParams}};

const STEPS: usize = 100;

fn main() {
    let sim_params = SimParams {
        particle_num: 10000,
        g: 0.000001,
        e: 0.0001,
        dt: 0.016,
    };
    println!("Initializing Simulation");
    let mut runner = pollster::block_on(OfflineHeadless::<NaiveSim>::new(sim_params, inits::disc_init)).unwrap();
    println!("Running Simulation");
    for _ in 0..STEPS {
        runner.step();
    }
    println!("Finished Running");
}
