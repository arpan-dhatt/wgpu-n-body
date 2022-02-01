use crate::sims::{Particle, SimParams};

use glam::Vec3A;
use rand::{distributions::Uniform, prelude::Distribution};

pub fn uniform_init(sim_params: &SimParams) -> Vec<Particle> {
    let mut rng = rand::thread_rng();
    let pos_unif = Uniform::new_inclusive(-1.0, 1.0);
    let mut initial_particles = Vec::with_capacity(sim_params.particle_num as usize);
    for _ in 0..sim_params.particle_num {
        initial_particles.push(Particle {
            position: [
                pos_unif.sample(&mut rng),
                pos_unif.sample(&mut rng),
                pos_unif.sample(&mut rng),
            ],
            velocity: [
                pos_unif.sample(&mut rng) * 0.001,
                pos_unif.sample(&mut rng) * 0.001,
                pos_unif.sample(&mut rng) * 0.001,
            ],
            acceleration: [0.0, 0.0, 0.0],
            ..Particle::default()
        });
    }
    initial_particles
}

pub fn disc_init(sim_params: &SimParams) -> Vec<Particle> {
    let coeff: f32 = 0.1;
    let mut rng = rand::thread_rng();
    let unif = Uniform::new_inclusive(-1.0, 1.0);
    let mut initial_particles = Vec::with_capacity(sim_params.particle_num as usize);
    for _ in 0..sim_params.particle_num {
        let mut pos: Vec3A = Vec3A::new(unif.sample(&mut rng), unif.sample(&mut rng), 0.0);
        while pos.length() > 1.0 {
            pos = Vec3A::new(unif.sample(&mut rng), unif.sample(&mut rng), 0.0);
        }
        let vel = coeff * 1.0 * (pos.length().sqrt() + 0.001) * pos.cross(Vec3A::Z).normalize();
        initial_particles.push(Particle {
            position: pos.to_array(),
            velocity: vel.to_array(),
            acceleration: [0.0, 0.0, 0.0],
            ..Particle::default()
        })
    }
    initial_particles
}
