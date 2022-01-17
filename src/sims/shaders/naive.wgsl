struct Triple {
    x: f32; y: f32; z: f32;
};

struct SimParams {
    num_particles: u32;
    g: f32;
    e: f32;
    dt: f32;
};

struct Triples {
    triples: [[stride(12)]] array<Triple>;
};

[[group(0), binding(0)]] var<uniform> params: SimParams;
[[group(0), binding(1)]] var<storage, read> particlesSrcPos: Triples;
[[group(0), binding(2)]] var<storage, read> particlesSrcVel: Triples;
[[group(0), binding(3)]] var<storage, read> particlesSrcAcc: Triples;
[[group(0), binding(4)]] var<storage, read_write> particlesDstPos: Triples;
[[group(0), binding(5)]] var<storage, read_write> particlesDstVel: Triples;
[[group(0), binding(6)]] var<storage, read_write> particlesDstAcc: Triples;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let total = arrayLength(&particlesSrcPos.triples);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    var aPos = vec3<f32>(
        particlesSrcPos.triples[index].x, 
        particlesSrcPos.triples[index].y,
        particlesSrcPos.triples[index].z
    );
    var aVel = vec3<f32>(
        particlesSrcVel.triples[index].x, 
        particlesSrcVel.triples[index].y,
        particlesSrcVel.triples[index].z
    );
    var aAcc = vec3<f32>(
        particlesSrcAcc.triples[index].x, 
        particlesSrcAcc.triples[index].y,
        particlesSrcAcc.triples[index].z
    );

    aVel = aVel + aAcc * params.dt / 2.0;
    aPos = aPos + aVel * params.dt;

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var i: u32 = 0u;
    loop {
        if (i >= total) {
            break;
        }
        if (i == index) {
            continue;
        }

        var bPos = vec3<f32>(
            particlesSrcPos.triples[i].x, 
            particlesSrcPos.triples[i].y,
            particlesSrcPos.triples[i].z
        );
        var bVel = vec3<f32>(
            particlesSrcVel.triples[i].x, 
            particlesSrcVel.triples[i].y,
            particlesSrcVel.triples[i].z
        );

        let r: f32 = distance(aPos, bPos);
        let force: vec3<f32> = params.g / (r * r * r + params.e) * normalize(bPos - aPos); 
        let _acc: vec3<f32> = force;
        acc = acc + _acc * params.dt;

        continuing {
            i = i + 1u;
        }
    }
    aVel = aVel + acc * params.dt / 2.0;

    particlesDstPos.triples[index] = Triple(aPos.x, aPos.y, aPos.z);
    particlesDstVel.triples[index] = Triple(aVel.x, aVel.y, aVel.z);
    particlesDstAcc.triples[index] = Triple(acc.x, acc.y, acc.z);
}
