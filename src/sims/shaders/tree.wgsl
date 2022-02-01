struct Octant {
    c: vec3<f32>;
    mass: f32;
    bodies: u32;
    children: array<u32,8>;
};

struct Particle {
    p: vec3<f32>;
    v: vec3<f32>;
    a: vec3<f32>;
};

struct SimParams {
    num_particles: u32;
    g: f32;
    e: f32;
    dt: f32;
};

struct TreeSimParams {
    theta: f32;
    root_width: f32;
};

struct Particles {
    particles: [[stride(48)]] array<Particle>;
};

struct Octants {
    octants: [[stride(64)]] array<Octant>;
};

[[group(0), binding(0)]] var<uniform> params: SimParams;
[[group(0), binding(1)]] var<uniform> tree_params: TreeSimParams;
[[group(0), binding(2)]] var<storage, read> particlesSrc: Particles;
[[group(0), binding(3)]] var<storage, read> treeSrc: Octants;
[[group(0), binding(4)]] var<storage, read_write> particlesDst: Particles;

fn getAcc(aPos: vec3<f32>, index: u32, total: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    // simulated recursive stack (quad index stack, node width stack)
    var oct_stack: array<u32, 64>;
    var size_stack: array<f32, 64>;
    // set root node to start
    oct_stack[0] = 0u;
    size_stack[0] = tree_params.root_width; 
    var size: u32 = 1u;
    loop {
        if (size == 0u) {
            break;
        }
        let top_oct: Octant = treeSrc.octants[ oct_stack[size - 1u] ];
        let top_size = size_stack[size - 1u];
        let cog = top_oct.c;
        let dist = distance(aPos, cog);
        if (dist < 0.00001 & top_oct.bodies == 1u) {
            // same body so skip calculation
            size = size - 1u;
            continue;
        }
        let sd = top_size / dist;
        if (sd < tree_params.theta) {
            // treat this as a single body since it's sufficiently far away
            let force: vec3<f32> = top_oct.mass * params.g / (dist * dist * dist + params.e) * normalize(cog - aPos);
            acc = acc + force * params.dt;
            size = size - 1u;
            continue;
        }
        // otherwise recurse further (after removing current frame)
        size = size - 1u;
        var curr_ix = oct_stack[size];
        var i: u32 = 0u;
        loop {
            if (i >= 8u) {
                break;
            }
            // add each subsection to the stack if it exists
            let child_ix = treeSrc.octants[curr_ix].children[i];
            if (child_ix != 0u) {
                size_stack[size] = top_size / 2.0;
                oct_stack[size] = child_ix;
                size = size + 1u;
            }
            i = i + 1u;
        }
    }
    return acc;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let total = arrayLength(&particlesSrc.particles);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    let _p = particlesSrc.particles[index];
    var aPos = _p.p;
    var aVel = _p.v;
    var aAcc = _p.a;

    aVel = aVel + aAcc * params.dt / 2.0;
    aPos = aPos + aVel * params.dt;
    let acc = getAcc(aPos, index, total);
    aVel = aVel + acc * params.dt / 2.0;

    particlesDst.particles[index] = Particle(aPos, aVel, aAcc);
}
