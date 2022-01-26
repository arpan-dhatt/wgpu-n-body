use crate::sims::Particle;
use parking_lot::Mutex;
use rayon::prelude::*;

/*pub struct Octree<'a> {
    root: Octant<'a>,
    pub bound: f32,
}

impl Octree<'_> {
    fn create_from(bodies: &[Particle]) -> Octree {
        let bound = bodies
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
        let par_bodies = bodies[1..].par_iter();
        let octree = Octree {
            root: Mutex::new(Some(OctantInner {
                cog: bodies[0].position,
                mass: 1.0,
                bodies: 1,
                children: [&Mutex::new(None); 8],
            })),
            bound,
        };
        par_bodies.for_each(|p| {
            let curr_node = &octree.root.lock();
            let mut node_center = [0.0; 3];
            let mut node_width = octree.bound * 2.0;
            while let Some(ref mut inner) = curr_node.map(|n| n) {
                if inner.bodies <= 1 {
                    break;
                }
                inner.bodies += 1;
                balance_cog(&mut inner.cog, inner.mass, &p.position);
                inner.mass += 1.0;
                let child_index = decide_octant(&node_center, &p.position);
                let child_octant = match &mut inner.children[child_index] {
                    Some(o) => {
                        0
                    },
                    None => {
                        0
                    }
                };
            }
        });
    }
}

type Octant<'a> = Mutex<Option<OctantInner<'a>>>;

#[derive(Clone, Debug)]
pub struct OctantInner<'a> {
    pub cog: [f32; 3],
    pub mass: f32,
    pub bodies: u32,
    pub children: [&'a Octant<'a>; 8],
}

#[inline]
fn decide_octant(center: &[f32; 3], point: &[f32; 3]) -> usize {
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
fn shift_node_center(node_center: &mut [f32; 3], node_width: &mut f32, child_octant: usize) {
    *node_width /= 2.0;
    node_center[0] += ((child_octant & 1) as i32 * 2 - 1) as f32 * *node_width / 2.0;
    node_center[1] += (((child_octant & 2) >> 1) as i32 * 2 - 1) as f32 * *node_width / 2.0;
    node_center[2] += (((child_octant & 4) >> 2) as i32 * 2 - 1) as f32 * *node_width / 2.0;
}*/
