use crate::sims::{Particles, SimParams};

use glam::Vec3A;
use rand::{distributions::Uniform, prelude::Distribution};

pub fn uniform_init(sim_params: &SimParams) -> Particles {
    let mut rng = rand::thread_rng();
    let pos_unif = Uniform::new_inclusive(-1.0, 1.0);
    let mut initial_particles = Particles::default();
    for _ in 0..sim_params.particle_num {
        initial_particles.position.push([
            pos_unif.sample(&mut rng),
            pos_unif.sample(&mut rng),
            pos_unif.sample(&mut rng),
        ]);
        initial_particles.velocity.push([
            pos_unif.sample(&mut rng) * 0.001,
            pos_unif.sample(&mut rng) * 0.001,
            pos_unif.sample(&mut rng) * 0.001,
        ]);
        initial_particles.acceleration.push([0.0, 0.0, 0.0]);
    }
    initial_particles
}

pub fn disc_init(sim_params: &SimParams) -> Particles {
    let coeff: f32 = 0.01;
    let mut rng = rand::thread_rng();
    let unif = Uniform::new_inclusive(-1.0, 1.0);
    let mut initial_particles = Particles::default();
    for _ in 0..sim_params.particle_num {
        let mut pos: Vec3A = Vec3A::new(unif.sample(&mut rng), unif.sample(&mut rng), 0.0);
        while pos.length() > 1.0 {
            pos = Vec3A::new(unif.sample(&mut rng), unif.sample(&mut rng), 0.0);
        }
        let vel = coeff * 1.0 / (pos.length().sqrt() + 0.001) * pos.cross(Vec3A::Z).normalize();
        initial_particles.position.push(pos.to_array());
        initial_particles.velocity.push(vel.to_array());
        initial_particles.acceleration.push([0.0, 0.0, 0.0]);

    }
    initial_particles
}
