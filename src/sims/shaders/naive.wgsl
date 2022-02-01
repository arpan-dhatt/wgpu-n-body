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

struct Particles {
    particles: [[stride(48)]] array<Particle>;
};

[[group(0), binding(0)]] var<uniform> params: SimParams;
[[group(0), binding(1)]] var<storage, read> particlesSrc: Particles;
[[group(0), binding(2)]] var<storage, read_write> particlesDst: Particles;

fn getAcc(aPos: vec3<f32>, index: u32, total: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var i: u32 = 0u;
    loop {
        if (i >= total) {
            break;
        }
        if (i == index) {
            continue;
        }

        let _q = particlesSrc.particles[i];
        var bPos = _q.p;
        var bVel = _q.v;

        let r: f32 = distance(aPos, bPos);
        let force: vec3<f32> = params.g / (r * r * r + params.e) * normalize(bPos - aPos); 
        let _acc: vec3<f32> = force;
        acc = acc + _acc * params.dt;

        continuing {
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
