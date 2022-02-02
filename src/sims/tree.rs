use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crossbeam_utils::thread;
use glam::Vec3A;
use image::EncodableLayout;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::utils::{slice_alloc::{Reserve, SliceAlloc}, self};

use super::{Particle, SimParams, Simulator};

pub struct TreeSim {
    sim_params: SimParams,
    tree_sim_params: TreeSimParams,
    tree_sim_params_buffer: wgpu::Buffer,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    particle_read_buffer: wgpu::Buffer,
    tree_buffer: wgpu::Buffer,
    tree_staging_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    work_group_count: u32,
    step_num: usize,
}

static NUM_CPUS: usize = 4;

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

        let tree_sim_params = TreeSimParams {
            theta: 0.5,
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
                    contents: unsafe {
                        let particle_slice = initial_particles.as_slice();
                        let num_bytes = particle_slice.len() * std::mem::size_of::<Particle>();
                        std::slice::from_raw_parts::<u8>(particle_slice.as_ptr() as *const _, num_bytes)
                    },
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
            tree_sim_params,
            tree_sim_params_buffer,
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

        let particle_read_data: &[Particle] = unsafe {
            utils::cast_slice(read_buffer_mapped.as_bytes())
        };
        let tree_staging_data: &mut [Octant] = unsafe {
            utils::cast_slice_mut(&mut tree_staging_mapped)
        };

        let octree_nodes = self.build_tree(
            particle_read_data,
            tree_staging_data,
            &queue,
            self.tree_sim_params.clone(),
        );

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
                (std::mem::size_of::<Octant>() as u32 * octree_nodes as u32) as _,
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

    fn cleanup(&mut self) {}
}

type BVec<'a, T> = bumpalo::collections::Vec<'a, T>;

#[derive(Debug)]
struct Partition<'a> {
    center: [f32; 3],
    width: f32,
    octant_ix: Option<Reserve<'a>>,
    particles_ix: Option<Vec<usize>>,
}

impl TreeSim {
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
                    position: Vec3A::ONE,
                    ..Particle::default()
                },
                |a, b| Particle {
                    position: a.position.max(b.position.abs()),
                    ..Particle::default()
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
        let (tx, rx) = crossbeam_channel::unbounded();
        // initialize slice allocator
        let mut tree_alloc = SliceAlloc::wrap(tree_data);
        let root_ix = tree_alloc.write(Octant::default());
        // create root partition (all particles)
        tx.send(Partition {
            center: [0.0; 3],
            width: bound[0] * 2.0,
            octant_ix: Some(root_ix),
            particles_ix: Some((0..particle_data.len()).collect()),
        })
        .expect("Failed to send Partition to MPMC Channel");
        // while there are partitions to process
        let active_threads = Arc::new(AtomicUsize::new(NUM_CPUS));
        thread::scope(|scope| {
            for _ in 0..NUM_CPUS {
                let ac = active_threads.clone();
                let txc = tx.clone();
                let rxc = rx.clone();
                let mut tac = tree_alloc.clone();
                scope.spawn(move |_| {
                    loop {
                        // process all available partitions
                        while let Ok(part) = rxc.try_recv() {
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
                            // temporary center of gravity variable to allow SIMD operations
                            let mut temp_cog = Vec3A::ZERO;
                            // calculate octant data and particle child subdivisions
                            for particle_ix in part.particles_ix.as_ref().unwrap() {
                                let p = particle_data[*particle_ix];
                                temp_cog += p.position;
                                octant.mass += 1.0;
                                let child_ix = Self::decide_octant(&part.center, &p.position);
                                if let Some(ref mut particles_ix) =
                                    child_partitions[child_ix].particles_ix
                                {
                                    // child particles list already exists
                                    particles_ix.push(*particle_ix);
                                } else {
                                    // needs to be created
                                    child_partitions[child_ix].particles_ix =
                                        Some(vec![*particle_ix]);
                                }
                            }
                            octant.bodies += part.particles_ix.unwrap().len() as u32;
                            temp_cog /= octant.mass;
                            temp_cog.write_to_slice(&mut octant.cog);
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
                                let mut child_octant = Octant::default();
                                let child_oct_handle = tac.write(child_octant.clone());
                                if part_count > 1 {
                                    // non-leaf node
                                    let child_oct_ix: usize = (&child_oct_handle).into();
                                    octant.children[i] = child_oct_ix as u32;
                                    child_part.octant_ix = Some(child_oct_handle);
                                    txc.send(child_part)
                                        .expect("Failed to send Child to MPMC Queue");
                                } else if part_count == 1 {
                                    // leaf node (complete octant processing and finish)
                                    particle_data[child_part.particles_ix.unwrap()[0]].position.write_to_slice(&mut child_octant.cog);
                                    child_octant.mass = 1.0;
                                    child_octant.bodies = 1;
                                    tac[child_oct_handle] = child_octant;
                                }
                            }
                            // write octant to array
                            tac[part.octant_ix.unwrap()] = octant;
                        }
                        if ac.fetch_sub(1, Ordering::Relaxed) <= 1 {
                            // all other threads are also idle, so processing must be finished
                            break;
                        }
                        ac.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        })
        .unwrap();
        tree_alloc.len()
    }

    #[inline]
    fn decide_octant(center: &[f32; 3], point: &Vec3A) -> usize {
        ((point[0] > center[0]) as usize)
            | (((point[1] > center[1]) as usize) << 1)
            | (((point[2] > center[2]) as usize) << 2)
    }

    #[inline]
    fn balance_cog(cog: &mut [f32; 3], curr_mass: f32, new_pos: &[f32; 3]) {
        cog[0] += (new_pos[0] - cog[0]) * (1.0 / (curr_mass + 1.0));
        cog[1] += (new_pos[1] - cog[1]) * (1.0 / (curr_mass + 1.0));
        cog[2] += (new_pos[2] - cog[2]) * (1.0 / (curr_mass + 1.0));
    }

    #[inline]
    fn shift_node_center(node_center: &[f32; 3], node_width: f32, child_octant: usize) -> [f32; 3] {
        [
            node_center[0] + ((child_octant & 1) as i32 * 2 - 1) as f32 * node_width / 4.0,
            node_center[1] + (((child_octant & 2) >> 1) as i32 * 2 - 1) as f32 * node_width / 4.0,
            node_center[2] + (((child_octant & 4) >> 2) as i32 * 2 - 1) as f32 * node_width / 4.0,
        ]
    }
}

/// Child Octant Positions:
/// ```
/// Front: -z   Back: +z
/// |---|---|   |---|---|
/// | 2 | 3 |   | 6 | 7 |
/// |---|---|   |---|---|
/// | 0 | 1 |   | 4 | 5 |
/// |---|---|   |---|---|
/// ```
#[repr(C)]
#[derive(Clone, Debug, Default)]
struct Octant {
    cog: [f32; 3],
    mass: f32,
    bodies: u32,
    children: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeSimParams {
    theta: f32,
    root_width: f32,
}
