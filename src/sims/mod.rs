mod naive;
mod tree;

pub use naive::NaiveSim;
pub use tree::TreeSim;

pub const PARTICLES_PER_GROUP: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
}

impl Particle {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
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
        init_fn: fn(&SimParams) -> Vec<Particle>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::CommandEncoder;
    fn dest_particle_slice(&self) -> wgpu::BufferSlice;
    fn sim_params(&self) -> SimParams;

    /// Optional Method that can be run while the GPU is executing code, helpful for resource
    /// cleanup
    fn cleanup(&mut self) {}
}
