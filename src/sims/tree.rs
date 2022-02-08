use std::{borrow::Cow, collections::VecDeque, ops::DerefMut, time::Instant};

use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::utils::slice_alloc::{Reserve, SliceAlloc};

use super::{Particle, SimParams, Simulator};

pub struct TreeSim {
    sim_params: SimParams,
    tree_sim_params: TreeSimParams,
    tree_sim_params_buffer: wgpu::Buffer,
    particle_bind_groups: Vec<wgpu::BindGroup>,
    particle_buffers: Vec<wgpu::Buffer>,
    particle_read_buffer: wgpu::Buffer,
    particle_write_buffer: wgpu::Buffer,
    tree_buffer: wgpu::Buffer,
    tree_staging_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    work_group_count: u32,
    step_num: usize,
    alloc_arena: bumpalo_herd::Herd,
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

        let tree_sim_params = TreeSimParams {
            theta: 0.75,
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
                                    * std::mem::size_of::<OctantRaw>())
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

        let particle_write_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Write Buffer"),
            size: (std::mem::size_of::<Particle>() as u32 * sim_params.particle_num) as _,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Completed Tree Buffer"),
            size: (std::mem::size_of::<OctantRaw>() as u32 * sim_params.particle_num * 4) as _,
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
            size: (std::mem::size_of::<OctantRaw>() as u32 * sim_params.particle_num * 4) as _,
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
            particle_write_buffer,
            tree_buffer,
            tree_staging_buffer,
            compute_pipeline,
            work_group_count,
            step_num: 0,
            alloc_arena: bumpalo_herd::Herd::new(),
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
        let write_buffer_slice = self.particle_write_buffer.slice(..);
        let tree_staging_slice = self.tree_staging_buffer.slice(..);

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
        let tree_staging_data: &mut [OctantRaw] =
            bytemuck::cast_slice_mut(&mut tree_staging_mapped);

        let now = Instant::now();
        let mut root_node = self.build_tree(particle_read_data, queue, self.tree_sim_params);
        println!("Tree Construction: {} µs", now.elapsed().as_micros());
        let now = Instant::now();
        Self::sort_particles_count_nodes(&mut root_node, particle_read_data, particle_write_data);
        println!(
            "Particle Sort and Node Count: {} µs ({} nodes)",
            now.elapsed().as_micros(),
            root_node.node_count
        );
        let now = Instant::now();
        Self::flatten_octree(&root_node, tree_staging_data, TraversalMode::PreOrder);
        println!("Octree Flattening: {} µs", now.elapsed().as_micros());

        drop(read_buffer_mapped);
        self.particle_read_buffer.unmap();
        drop(write_buffer_mapped);
        self.particle_write_buffer.unmap();
        drop(tree_staging_mapped);
        self.tree_staging_buffer.unmap();

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

        encoder.push_debug_group("flush tree staging buffer");
        {
            encoder.copy_buffer_to_buffer(
                &self.tree_staging_buffer,
                0,
                &self.tree_buffer,
                0,
                (std::mem::size_of::<OctantRaw>() as u32 * root_node.node_count as u32) as _,
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
        self.sim_params
    }

    fn cleanup(&mut self) {
        self.alloc_arena.reset();
    }
}

type BVec<'a, T> = bumpalo::collections::Vec<'a, T>;

#[derive(Debug)]
struct Partition<'a, 'b> {
    center: [f32; 3],
    width: f32,
    octant: Option<&'a mut OctantNode>,
    particles_ix: Option<BVec<'b, usize>>,
}

impl TreeSim {
    fn build_tree(
        &self,
        particle_data: &[Particle],
        queue: &wgpu::Queue,
        mut tree_sim_params: TreeSimParams,
    ) -> OctantNode {
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
        // create root node
        let mut root = OctantNode::default();
        let herd_member = self.alloc_arena.get();
        let member_bump = herd_member.as_bump();
        let mut part_queue = VecDeque::new();
        // create root partition (all particles)
        part_queue.push_back(Partition {
            center: [0.0; 3],
            width: bound[0] * 2.0,
            octant: Some(&mut root),
            particles_ix: Some(BVec::from_iter_in(0..particle_data.len(), member_bump)),
        });
        // while there are partitions to process
        while let Some(part) = part_queue.pop_front() {
            // create all possible child partitions (not always added to queue)
            let mut child_partitions: Vec<Partition> = (0..8)
                .map(|ix| Partition {
                    center: Self::shift_node_center(&part.center, part.width, ix),
                    width: part.width / 2.0,
                    octant: None,
                    particles_ix: None,
                })
                .collect();
            // partition's octant (to be assigned to correct reference later)
            let mut octant = part.octant.unwrap();
            // calculate octant data and particle child subdivisions on the stack
            let mut cog = [0.0; 3];
            let mut mass = 0.0;
            for particle_ix in part.particles_ix.as_ref().unwrap() {
                let p = particle_data[*particle_ix];
                cog[0] += p.position[0];
                cog[1] += p.position[1];
                cog[2] += p.position[2];
                mass += p.mass;
                let child_ix = Self::decide_octant(&part.center, &p.position);
                if let Some(ref mut particles_ix) = child_partitions[child_ix].particles_ix {
                    // child particles list already exists
                    particles_ix.push(*particle_ix);
                } else {
                    // needs to be created
                    child_partitions[child_ix].particles_ix =
                        Some(BVec::from_iter_in(Some(*particle_ix), member_bump));
                }
            }
            cog[0] /= mass;
            cog[1] /= mass;
            cog[2] /= mass;
            // assign finalized values to heap-allocated node
            octant.cog = cog;
            octant.mass = mass;
            octant.bodies += part.particles_ix.unwrap().len() as u32;
            // only add new partitions if non-leaf node
            for (mut child_part, child_ref) in
                child_partitions.into_iter().zip(octant.children.iter_mut())
            {
                let part_count = child_part
                    .particles_ix
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0);
                // zero-node does nothing
                if part_count == 0 {
                    continue;
                }
                match part_count {
                    1 => {
                        // leaf node (complete octant processing and finish)
                        let leaf_particle =
                            particle_data[child_part.particles_ix.as_ref().unwrap()[0]];
                        let leaf_octant = OctantNode {
                            cog: leaf_particle.position,
                            mass: leaf_particle.mass,
                            bodies: 1,
                            // set first child to particle index for sorting particles by locality
                            one_body: child_part.particles_ix.unwrap()[0],
                            ..Default::default()
                        };
                        *child_ref = Some(Box::new(leaf_octant));
                    }
                    _ => {
                        // non-leaf node
                        *child_ref = Some(Box::new(OctantNode::default()));
                        child_part.octant = Some(child_ref.as_mut().unwrap().deref_mut());
                        part_queue.push_back(child_part);
                    }
                };
            }
        }
        root
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

    /// Wrapper for [`sort_particles_count_nodes_recursive`] function.
    fn sort_particles_count_nodes(
        root: &mut OctantNode,
        particles_src: &[Particle],
        particles_dst: &mut [Particle],
    ) {
        root.node_count =
            Self::sort_particles_count_nodes_recursive(root, particles_src, particles_dst);
    }

    /// Sorts particles according to an in-order traversal of the octree and counts the number of
    /// nodes in a subtree with the given node the root
    fn sort_particles_count_nodes_recursive(
        node: &mut OctantNode,
        particles_src: &[Particle],
        particles_dst: &mut [Particle],
    ) -> usize {
        if node.bodies == 1 {
            particles_dst[0] = particles_src[node.one_body];
            node.node_count = 1;
            return 1;
        } else {
            let mut slices = vec![];
            let mut remaining = particles_dst;
            for child_node in node.children.iter_mut() {
                if let Some(child_node) = child_node.as_mut() {
                    let (a_slice, b_slice) = remaining.split_at_mut(child_node.bodies as usize);
                    remaining = b_slice;
                    slices.push((child_node, a_slice));
                }
            }
            let num_descendants: usize = slices
                .into_par_iter()
                .map(|(child_node, child_slice)| {
                    Self::sort_particles_count_nodes_recursive(
                        child_node,
                        particles_src,
                        child_slice,
                    )
                })
                .sum();
            node.node_count = num_descendants + 1;
            return node.node_count;
        }
    }

    /// Places an octree into a raw octant array, returning the number of spots in the
    /// [`OctantRaw`] slice were used to write the given node and its descendants.
    ///
    /// # Arguments
    ///
    /// * `node` - root node of the octree subtree
    /// * `tree_dst` - sub-slice of original raw slice to place tree node into
    /// * `traversal` - the method in which to place nodes into the tree
    fn flatten_octree(node: &OctantNode, tree_dst: &mut [OctantRaw], traversal: TraversalMode) {
        match traversal {
            TraversalMode::LevelOrder => Self::flatten_octree_level_order(node, tree_dst, 0),
            TraversalMode::PreOrder => Self::flatten_octree_pre_order(node, tree_dst, 0),
        }
    }

    fn flatten_octree_level_order(node: &OctantNode, tree_dst: &mut [OctantRaw], offset: u32) {}

    /// Parallelized placement of octree nodes into slice in pre-order
    fn flatten_octree_pre_order(node: &OctantNode, tree_dst: &mut [OctantRaw], offset: usize) {
        // convert most octant node data to raw format
        let mut raw: OctantRaw = node.into();
        if raw.bodies == 1 {
            tree_dst[0] = raw;
            return;
        }
        let mut slices = vec![];
        // reserve one spot for subtree root node
        let (tree_dst, mut remaining) = tree_dst.split_at_mut(1);
        // set global offset for child
        let mut child_offset = offset + 1;
        for child_node in node.children.iter() {
            if let Some(child_node) = child_node.as_ref() {
                let (a_slice, b_slice) = remaining.split_at_mut(child_node.node_count);
                remaining = b_slice;
                slices.push(Some((child_node, a_slice, child_offset)));
                child_offset += child_node.node_count;
            } else {
                slices.push(None)
            }
        }
        slices.into_par_iter().zip(&mut raw.children).for_each(|(opt, child_idx)| {
            if let Some((child_node, child_slice, child_offset)) = opt {
                // assign global child idx to raw octant
                *child_idx = child_offset as u32;
                Self::flatten_octree_pre_order(&child_node, child_slice, child_offset);
            }
        });
        tree_dst[0] = raw;
    }
}

enum TraversalMode {
    LevelOrder,
    PreOrder,
}

impl From<&OctantNode> for OctantRaw {
    fn from(o: &OctantNode) -> Self {
        OctantRaw {
            cog: o.cog,
            mass: o.mass,
            bodies: o.bodies,
            children: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct OctantRaw {
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

#[derive(Clone, Debug, Default)]
struct OctantNode {
    cog: [f32; 3],
    mass: f32,
    bodies: u32,
    node_count: usize,
    children: [Option<Box<OctantNode>>; 8],
    // 0 unless bodies == 1, then used to indicate body index in particles array for sorting
    one_body: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeSimParams {
    theta: f32,
    root_width: f32,
}
