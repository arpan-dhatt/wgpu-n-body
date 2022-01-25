use std::sync::atomic::AtomicBool;

pub struct Octree {
    storage: Vec<Octant>,
    locks: Vec<AtomicBool>
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Octant {
    pub cog: [f32; 3],
    pub mass: f32,
    pub bodies: u32,
    pub children: [u32; 8]
}

impl Octant {
    
}
