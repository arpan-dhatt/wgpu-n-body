mod naive;

pub use naive::NaiveSim;

pub const PARTICLES_PER_GROUP: u32 = 64;

pub struct Particles {
    pub position: Vec<[f32; 3]>,
    pub velocity: Vec<[f32; 3]>,
    pub acceleration: Vec<[f32; 3]>,
}

impl Default for Particles {
    fn default() -> Self {
        Particles {
            position: Vec::new(),
            velocity: Vec::new(),
            acceleration: Vec::new(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub particle_num: u32,
    pub g: f32,
    pub e: f32,
    pub dt: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            particle_num: 10000,
            g: 0.000001,
            e: 0.0001,
            dt: 0.016,
        }
    }
}

pub trait Simulator {
    fn new(
        device: &wgpu::Device,
        sim_params: SimParams,
        init_fn: fn(&SimParams) -> Particles,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn encode(&mut self, encoder: &mut wgpu::CommandEncoder);
    fn dest_particle_slice(&self) -> wgpu::BufferSlice;
    fn sim_params(&self) -> SimParams;
}
