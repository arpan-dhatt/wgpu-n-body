struct Particle {
    px: f32; py: f32; pz: f32;
    vx: f32; vy: f32; vz: f32;
};

struct SimParams {
    num_particles: u32;
    g: f32;
    e: f32;
    dt: f32;
};

struct Particles {
    particles: [[stride(24)]] array<Particle>;
};

[[group(0), binding(0)]] var<uniform> params: SimParams;
[[group(0), binding(1)]] var<storage, read> particlesSrc: Particles;
[[group(0), binding(2)]] var<storage, read_write> particlesDst: Particles;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let total = arrayLength(&particlesSrc.particles);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    let _p = particlesSrc.particles[index];
    var aPos = vec3<f32>(_p.px, _p.py, _p.pz);
    var aVel = vec3<f32>(_p.vx, _p.vy, _p.vz);

    var i: u32 = 0u;
    loop {
        if (i >= total) {
            break;
        }
        if (i == index) {
            continue;
        }

        let _q = particlesSrc.particles[i];
        var bPos = vec3<f32>(_q.px, _q.py, _q.pz);
        var bVel = vec3<f32>(_q.vx, _q.vy, _q.vz);

        let r: f32 = distance(aPos, bPos);
        let force: vec3<f32> = params.g / (r * r * r + params.e) * normalize(bPos - aPos); 
        let acc: vec3<f32> = force;
        aVel = aVel + acc * params.dt;

        continuing {
            i = i + 1u;
        }
    }
    aPos = aPos + aVel * params.dt;

    particlesDst.particles[index] = Particle(aPos.x, aPos.y, aPos.z, aVel.x, aVel.y, aVel.z);
}
