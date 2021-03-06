use std::{borrow::Cow, collections::VecDeque};

use log::warn;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::utils::slice_alloc::{Reserve, SliceAlloc};

use super::{AddParams, Particle, SimParams, Simulator};

pub struct TreeSim {
    sim_params: SimParams,
    tree_sim_params: TreeSimParams,
    tree_sim_params_buffer: wgpu::Buffer,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    particle_read_buffer: Option<wgpu::Buffer>,
    particle_write_buffer: wgpu::Buffer,
    tree_buffer: wgpu::Buffer,
    tree_staging_buffer: Option<wgpu::Buffer>,
    compute_pipeline: wgpu::ComputePipeline,
    work_group_count: u32,
    step_num: usize,
    mappable_primary_buffers: bool,
    alloc_arena: bumpalo::Bump,
}

impl Simulator for TreeSim {
    fn new(
        device: &wgpu::Device,
        sim_params: SimParams,
        add_params: AddParams,
        mappable_primary_buffers: bool,
        init_fn: fn(&SimParams) -> Vec<Particle>,
    ) -> anyhow::Result<Self> {
        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params Buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tree_sim_params = TreeSimParams {
            theta: match add_params {
                AddParams::TreeSimParams { theta } => theta,
                _ => {
                    warn!("No Theta Value Provided, using default: 0.75");
                    0.75
                }
            },
            root_width: 2.0,
        };
        let tree_sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tree Sim Specific Params"),
            contents: bytemuck::cast_slice(&[tree_sim_params]),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<TreeSimParams>() as _,
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
                                (sim_params.particle_num as usize * std::mem::size_of::<Particle>())
                                    as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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
                        binding: 4,
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
                        | wgpu::BufferUsages::COPY_DST
                        | match mappable_primary_buffers {
                            true => wgpu::BufferUsages::MAP_READ,
                            false => wgpu::BufferUsages::empty(),
                        },
                }),
            );
        }

        let particle_read_buffer = if !mappable_primary_buffers {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Read Buffer"),
                size: (std::mem::size_of::<Particle>() as u32 * sim_params.particle_num) as _,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let particle_write_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Write Buffer"),
            size: (std::mem::size_of::<Particle>() as u32 * sim_params.particle_num) as _,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Completed Tree Buffer"),
            size: (std::mem::size_of::<Octant>() as u32 * sim_params.particle_num * 4) as _,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | match mappable_primary_buffers {
                    true => wgpu::BufferUsages::MAP_WRITE,
                    false => wgpu::BufferUsages::empty(),
                },
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
                        resource: tree_sim_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tree_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(),
                    },
                ],
            }));
        }

        let tree_staging_buffer = if !mappable_primary_buffers {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tree Staging Buffer"),
                size: (std::mem::size_of::<Octant>() as u32 * sim_params.particle_num * 4) as _,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let work_group_count =
            ((sim_params.particle_num as f32) / (super::PARTICLES_PER_GROUP as f32)).ceil() as u32;

        Ok(Self {
            sim_params,
            tree_sim_params,
            tree_sim_params_buffer,
            particle_bind_groups,
            particle_buffers,
            particle_read_buffer,
            particle_write_buffer,
            tree_buffer,
            tree_staging_buffer,
            compute_pipeline,
            work_group_count,
            step_num: 0,
            mappable_primary_buffers,
            alloc_arena: bumpalo::Bump::new(),
        })
    }

    fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::CommandEncoder {
        if self.step_num == 0 {
            // empty command is sent to the queue to make initial mapping see particles
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Initial Map Update"),
            });
            queue.submit([encoder.finish()]);
        }

        let read_buffer_slice = self.get_particle_read_slice(device, queue);
        let write_buffer_slice = self.particle_write_buffer.slice(..);
        let tree_staging_slice = self.get_tree_write_slice(device, queue);
        let read_buffer_future = read_buffer_slice.map_async(wgpu::MapMode::Read);
        let write_buffer_future = write_buffer_slice.map_async(wgpu::MapMode::Write);
        let tree_staging_future = tree_staging_slice.map_async(wgpu::MapMode::Write);
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(read_buffer_future).unwrap();
        pollster::block_on(write_buffer_future).unwrap();
        pollster::block_on(tree_staging_future).unwrap();
        let read_buffer_mapped = read_buffer_slice.get_mapped_range();
        let mut write_buffer_mapped = write_buffer_slice.get_mapped_range_mut();
        let mut tree_staging_mapped = tree_staging_slice.get_mapped_range_mut();

        let particle_read_data: &[Particle] = bytemuck::cast_slice(&read_buffer_mapped);
        let particle_write_data: &mut [Particle] =
            bytemuck::cast_slice_mut(&mut write_buffer_mapped);
        let tree_staging_data: &mut [Octant] = bytemuck::cast_slice_mut(&mut tree_staging_mapped);

        let octree_nodes = self.build_tree(
            particle_read_data,
            tree_staging_data,
            queue,
            self.tree_sim_params,
        );

        Self::sort_particles(particle_read_data, particle_write_data, tree_staging_data);

        drop(write_buffer_mapped);
        drop(read_buffer_mapped);
        drop(tree_staging_mapped);
        self.particle_write_buffer.unmap();
        if self.mappable_primary_buffers {
            self.particle_buffers[self.step_num % 2].unmap();
            self.tree_buffer.unmap();
        } else {
            self.particle_read_buffer.as_ref().unwrap().unmap();
            self.tree_staging_buffer.as_ref().unwrap().unmap();
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tree Flush/Compute/Render Command"),
        });

        encoder.push_debug_group("flush sorted particle buffer");
        {
            encoder.copy_buffer_to_buffer(
                &self.particle_write_buffer,
                0,
                &self.particle_buffers[self.step_num % 2],
                0,
                (std::mem::size_of::<Particle>() as u32 * self.sim_params.particle_num as u32) as _,
            );
        }
        encoder.pop_debug_group();

        if !self.mappable_primary_buffers {
            encoder.push_debug_group("flush tree staging buffer");
            {
                encoder.copy_buffer_to_buffer(
                    &self.tree_staging_buffer.as_ref().unwrap(),
                    0,
                    &self.tree_buffer,
                    0,
                    (std::mem::size_of::<Octant>() as u32 * octree_nodes as u32) as _,
                );
            }
            encoder.pop_debug_group();
        }

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
        self.sim_params
    }

    fn cleanup(&mut self) {
        self.alloc_arena.reset();
    }
}

type BVec<'a, T> = bumpalo::collections::Vec<'a, T>;

#[derive(Debug)]
struct Partition<'a> {
    center: [f32; 3],
    width: f32,
    octant_ix: Option<Reserve<'a>>,
    particles_ix: Option<BVec<'a, usize>>,
}

impl TreeSim {
    fn get_particle_read_slice(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::BufferSlice {
        if self.mappable_primary_buffers {
            // buffer copy is unnecessary
            self.particle_buffers[self.step_num % 2].slice(..)
        } else {
            let mut read_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Particle Data Reader Command"),
            });
            {
                read_encoder.copy_buffer_to_buffer(
                    &self.particle_buffers[self.step_num % 2],
                    0,
                    &self.particle_read_buffer.as_ref().unwrap(),
                    0,
                    (std::mem::size_of::<Particle>() as u32 * self.sim_params.particle_num) as _,
                );
            }
            queue.submit(Some(read_encoder.finish()));
            self.particle_read_buffer.as_ref().unwrap().slice(..)
        }
    }

    fn get_tree_write_slice(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> wgpu::BufferSlice {
        if self.mappable_primary_buffers {
            self.tree_buffer.slice(..)
        } else {
            self.tree_staging_buffer.as_ref().unwrap().slice(..)
        }
    }

    fn build_tree(
        &self,
        particle_data: &[Particle],
        tree_data: &mut [Octant],
        queue: &wgpu::Queue,
        mut tree_sim_params: TreeSimParams,
    ) -> usize {
        let bound = particle_data
            .par_iter()
            .cloned()
            .reduce(
                || Particle {
                    position: [1.0; 3],
                    velocity: [0.0; 3],
                    acceleration: [0.0; 3],
                    mass: 1.0,
                },
                |a, b| Particle {
                    position: [
                        a.position[0].abs().max(b.position[0].abs()),
                        a.position[1].abs().max(b.position[1].abs()),
                        a.position[2].abs().max(b.position[2].abs()),
                    ],
                    velocity: [0.0; 3],
                    acceleration: [0.0; 3],
                    mass: 1.0,
                },
            )
            .position;
        let bound = bound[0].max(bound[1]).max(bound[2]);
        // write new root bounds data for gpu force calculation
        tree_sim_params = TreeSimParams {
            theta: tree_sim_params.theta,
            root_width: bound * 2.0,
        };
        queue.write_buffer(
            &self.tree_sim_params_buffer,
            0,
            bytemuck::cast_slice(&[tree_sim_params]),
        );
        let bound = [bound; 3];
        let mut part_queue = VecDeque::new();
        // initialize slice allocator
        let mut tree_alloc = SliceAlloc::wrap(tree_data);
        let root_ix = tree_alloc.write(Octant::default());
        // create root partition (all particles)
        part_queue.push_back(Partition {
            center: [0.0; 3],
            width: bound[0] * 2.0,
            octant_ix: Some(root_ix),
            particles_ix: Some(BVec::from_iter_in(
                0..particle_data.len(),
                &self.alloc_arena,
            )),
        });
        // while there are partitions to process
        while let Some(part) = part_queue.pop_front() {
            // create all possible child partitions (not always added to queue)
            let mut child_partitions: Vec<Partition> = (0..8)
                .map(|ix| Partition {
                    center: Self::shift_node_center(&part.center, part.width, ix),
                    width: part.width / 2.0,
                    octant_ix: None,
                    particles_ix: None,
                })
                .collect();
            // partition's octant
            let mut octant = Octant::default();
            // calculate octant data and particle child subdivisions
            for particle_ix in part.particles_ix.as_ref().unwrap() {
                let p = particle_data[*particle_ix];
                octant.cog[0] += p.position[0] * p.mass;
                octant.cog[1] += p.position[1] * p.mass;
                octant.cog[2] += p.position[2] * p.mass;
                octant.mass += p.mass;
                let child_ix = Self::decide_octant(&part.center, &p.position);
                if let Some(ref mut particles_ix) = child_partitions[child_ix].particles_ix {
                    // child particles list already exists
                    particles_ix.push(*particle_ix);
                } else {
                    // needs to be created
                    child_partitions[child_ix].particles_ix =
                        Some(BVec::from_iter_in(Some(*particle_ix), &self.alloc_arena));
                }
            }
            octant.bodies += part.particles_ix.unwrap().len() as u32;
            octant.cog[0] /= octant.mass;
            octant.cog[1] /= octant.mass;
            octant.cog[2] /= octant.mass;
            // only add new partitions if non-leaf node
            for (i, mut child_part) in child_partitions.into_iter().enumerate() {
                let part_count = child_part
                    .particles_ix
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0);
                // zero-node does nothing
                if part_count == 0 {
                    continue;
                }
                let child_oct_handle = tree_alloc.write(Octant::default());
                let child_oct_ix: usize = (&child_oct_handle).into();
                octant.children[i] = child_oct_ix as u32;
                match part_count {
                    1 => {
                        // leaf node (complete octant processing and finish)
                        let leaf_particle =
                            particle_data[child_part.particles_ix.as_ref().unwrap()[0]];
                        let mut leaf_octant = Octant {
                            cog: leaf_particle.position,
                            mass: leaf_particle.mass,
                            bodies: 1,
                            ..Default::default()
                        };
                        // set first child to particle index for sorting particles by locality
                        leaf_octant.children[0] = child_part.particles_ix.unwrap()[0] as u32;
                        tree_alloc[child_oct_handle] = leaf_octant;
                    }
                    _ => {
                        // non-leaf node
                        child_part.octant_ix = Some(child_oct_handle);
                        part_queue.push_back(child_part);
                    }
                };
            }
            // write octant to array
            tree_alloc[part.octant_ix.unwrap()] = octant;
        }
        tree_alloc.len()
    }

    #[inline]
    fn decide_octant(center: &[f32; 3], point: &[f32; 3]) -> usize {
        ((point[0] > center[0]) as usize)
            | (((point[1] > center[1]) as usize) << 1)
            | (((point[2] > center[2]) as usize) << 2)
    }

    #[inline]
    fn shift_node_center(node_center: &[f32; 3], node_width: f32, child_octant: usize) -> [f32; 3] {
        [
            node_center[0] + ((child_octant & 1) as i32 * 2 - 1) as f32 * node_width / 4.0,
            node_center[1] + (((child_octant & 2) >> 1) as i32 * 2 - 1) as f32 * node_width / 4.0,
            node_center[2] + (((child_octant & 4) >> 2) as i32 * 2 - 1) as f32 * node_width / 4.0,
        ]
    }

    fn sort_particles(
        particles_src: &[Particle],
        particles_dst: &mut [Particle],
        tree_data: &[Octant],
    ) {
        Self::sort_particles_recursive(tree_data[0], particles_src, particles_dst, tree_data);
    }

    fn sort_particles_recursive(
        octant: Octant,
        particles_src: &[Particle],
        particles_dst: &mut [Particle],
        tree_data: &[Octant],
    ) {
        if octant.bodies == 1 {
            particles_dst[0] = particles_src[octant.children[0] as usize];
        } else {
            let mut slices = vec![];
            let mut remaining = particles_dst;
            for child_ix in octant.children {
                if child_ix != 0 {
                    let child_octant = tree_data[child_ix as usize];
                    let (a_slice, b_slice) = remaining.split_at_mut(child_octant.bodies as usize);
                    remaining = b_slice;
                    slices.push((child_octant, a_slice));
                }
            }
            slices
                .par_iter_mut()
                .for_each(|(child_octant, child_slice)| {
                    Self::sort_particles_recursive(
                        *child_octant,
                        particles_src,
                        child_slice,
                        tree_data,
                    );
                });
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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
    // if bodies == 1 then read data from particles array (first child ix)
    bodies: u32,
    children: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeSimParams {
    theta: f32,
    root_width: f32,
}
