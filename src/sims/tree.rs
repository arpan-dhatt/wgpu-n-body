use std::{borrow::Cow, cmp};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use wgpu::util::DeviceExt;

use super::{Particle, SimParams, Simulator};

pub struct TreeSim {
    sim_params: SimParams,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    particle_read_buffer: wgpu::Buffer,
    tree_buffer: wgpu::Buffer,
    tree_staging_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    work_group_count: u32,
    step_num: usize,
}

impl Simulator for TreeSim {
    fn new(
        device: &wgpu::Device,
        sim_params: SimParams,
        init_fn: fn(&SimParams) -> Vec<super::Particle>,
    ) -> anyhow::Result<Self> {
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params Buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let compute_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Compute Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/tree.wgsl"))),
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<SimParams>() as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (sim_params.particle_num as usize * std::mem::size_of::<Particle>())
                                    as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (sim_params.particle_num as usize
                                    * 4
                                    * std::mem::size_of::<Octant>())
                                    as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (sim_params.particle_num as usize * std::mem::size_of::<Particle>())
                                    as _,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_module,
            entry_point: "main",
        });

        let initial_particles = init_fn(&sim_params);

        let mut particle_buffers = Vec::<wgpu::Buffer>::new();
        let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            particle_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Particle Buffer {}", i)),
                    contents: bytemuck::cast_slice(&initial_particles),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }),
            );
        }

        let particle_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Read Buffer"),
            size: (std::mem::size_of::<Particle>() as u32 * sim_params.particle_num) as _,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Completed Tree Buffer"),
            size: (std::mem::size_of::<Octant>() as u32 * sim_params.particle_num * 4) as _,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for i in 0..2 {
            particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bind Group {}", i)),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tree_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(),
                    },
                ],
            }));
        }

        let tree_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tree Staging Buffer"),
            size: (std::mem::size_of::<Octant>() as u32 * sim_params.particle_num * 4) as _,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let work_group_count =
            ((sim_params.particle_num as f32) / (super::PARTICLES_PER_GROUP as f32)).ceil() as u32;

        Ok(Self {
            sim_params,
            particle_bind_groups,
            particle_buffers,
            particle_read_buffer,
            tree_buffer,
            tree_staging_buffer,
            compute_pipeline,
            work_group_count,
            step_num: 0,
        })
    }

    fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::CommandEncoder {
        let mut read_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Particle Data Reader Command"),
        });
        {
            read_encoder.copy_buffer_to_buffer(
                &self.particle_buffers[self.step_num % 2],
                0,
                &self.particle_read_buffer,
                0,
                (std::mem::size_of::<Particle>() as u32 * self.sim_params.particle_num) as _,
            );
        }
        queue.submit(Some(read_encoder.finish()));

        let read_buffer_slice = self.particle_read_buffer.slice(..);
        let tree_staging_slice = self.tree_staging_buffer.slice(..);

        let read_buffer_future = read_buffer_slice.map_async(wgpu::MapMode::Read);
        let tree_staging_future = tree_staging_slice.map_async(wgpu::MapMode::Write);
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(read_buffer_future).unwrap();
        pollster::block_on(tree_staging_future).unwrap();

        let read_buffer_mapped = read_buffer_slice.get_mapped_range();
        let mut tree_staging_mapped = tree_staging_slice.get_mapped_range_mut();

        let particle_read_data: &[Particle] = bytemuck::cast_slice(&read_buffer_mapped);
        let tree_staging_data: &mut [Octant] = bytemuck::cast_slice_mut(&mut tree_staging_mapped);

        self.build_tree(particle_read_data, tree_staging_data);

        drop(read_buffer_mapped);
        self.particle_read_buffer.unmap();
        drop(tree_staging_mapped);
        self.tree_staging_buffer.unmap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tree Flush/Compute/Render Command"),
        });

        encoder.push_debug_group("flush tree staging buffer");
        {
            encoder.copy_buffer_to_buffer(
                &self.tree_staging_buffer,
                0,
                &self.tree_buffer,
                0,
                (std::mem::size_of::<Octant>() as u32 * self.sim_params.particle_num * 4) as _,
            );
        }
        encoder.pop_debug_group();

        encoder.push_debug_group("n-body movement");
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.particle_bind_groups[self.step_num % 2], &[]);
            cpass.dispatch(self.work_group_count, 1, 1);
        }
        encoder.pop_debug_group();
        self.step_num += 1;

        encoder
    }

    fn dest_particle_slice(&self) -> wgpu::BufferSlice {
        self.particle_buffers[(self.step_num + 1) % 2].slice(..)
    }

    fn sim_params(&self) -> SimParams {
        self.sim_params.clone()
    }
}

impl TreeSim {
    fn build_tree(&self, particle_data: &[Particle], tree_data: &mut [Octant]) {
        let bound = particle_data
            .par_iter()
            .cloned()
            .reduce(
                || Particle {
                    position: [1.0; 3],
                    velocity: [0.0; 3],
                    acceleration: [0.0; 3],
                },
                |a, b| Particle {
                    position: [
                        a.position[0].abs().max(b.position[0].abs()),
                        a.position[1].abs().max(b.position[1].abs()),
                        a.position[2].abs().max(b.position[2].abs()),
                    ],
                    velocity: [0.0; 3],
                    acceleration: [0.0; 3],
                },
            )
            .position;
        let bound = bound[0].max(bound[1]).max(bound[2]);
        let bound = [bound; 3];
        tree_data[0] = Octant {
            cog: particle_data[0].position,
            mass: 1.0,
            bodies: 1,
            children: [0; 8],
        };
        let mut alloced_nodes = 1;
        for particle in &particle_data[1..] {
            let mut node_ix = 0;
            let mut node_center = [0.0; 3];
            let mut node_width = bound[0] * 2.0;
            // find insertion point
            while tree_data[node_ix].bodies > 1 {
                tree_data[node_ix].bodies += 1;
                tree_data[node_ix].mass += 1.0;
                Self::balance_cog(&mut tree_data[node_ix].cog, tree_data[node_ix].mass, &particle.position);
                let child_octant = Self::decide_octant(&node_center, &particle.position);
                if tree_data[node_ix].children[child_octant] == 0 {
                    // if child doesn't exist, needs to be created
                    tree_data[alloced_nodes] = Octant {
                        cog: [0.0; 3],
                        mass: 0.0,
                        bodies: 0,
                        children: [0; 8],
                    };
                    node_ix = alloced_nodes;
                    alloced_nodes += 1;
                } else {
                    // child exists, so continue iterating
                    node_ix = tree_data[node_ix].children[child_octant] as usize;
                    node_width /= 2.0;
                    // shift node center
                    node_center[0] += (( child_octant | 1 ) * 2 - 1) as f32 * node_width / 2.0;
                    node_center[1] += (( child_octant | 2 ) * 2 - 1) as f32 * node_width / 2.0;
                    node_center[2] += (( child_octant | 4 ) * 2 - 1) as f32 * node_width / 2.0;
                }
            }
            // insert particle
            if tree_data[node_ix].bodies == 0 {
                // empty node just for this particle
                tree_data[node_ix].cog = particle.position;
                tree_data[node_ix].mass += 1.0;
                tree_data[node_ix].bodies = 1;
            } else {
                // non-empty node (1 body, possibly requiring numerous subdivisions)
                tree_data[node_ix].bodies += 1;
                tree_data[node_ix].mass += 1.0;
                Self::balance_cog(&mut tree_data[node_ix].cog, tree_data[node_ix].mass, &particle.position);

            }
        }
    }

    #[inline]
    fn decide_octant(center: &[f32; 3], point: &[f32; 3]) -> usize {
        ((point[0] > center[0]) as usize)
            | (((point[1] > center[1]) as usize) << 1)
            | (((point[2] > center[2]) as usize) << 2)
    }

    #[inline]
    fn balance_cog(cog: &mut [f32; 3], curr_mass: f32, new_pos: &[f32; 3]) {
        cog[0] += (new_pos[0] - cog[0]) * ( 1.0 / (curr_mass + 1.0));
        cog[1] += (new_pos[1] - cog[1]) * ( 1.0 / (curr_mass + 1.0));
        cog[2] += (new_pos[2] - cog[2]) * ( 1.0 / (curr_mass + 1.0));
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Octant {
    /// Child Octant Positions:
    /// ```
    /// Front: -z   Back: +z
    /// |---|---|   |---|---|
    /// | 2 | 3 |   | 6 | 7 |
    /// |---|---|   |---|---|
    /// | 0 | 1 |   | 4 | 5 |
    /// |---|---|   |---|---|
    /// ```
    cog: [f32; 3],
    mass: f32,
    bodies: u32,
    children: [u32; 8],
}
