[[stage(vertex)]]
fn main_vs(
    [[location(0)]] particle_pos: vec3<f32>,
    [[location(1)]] particle_vel: vec3<f32>,
    [[location(2)]] position: vec2<f32>,
) -> [[builtin(position)]] vec4<f32> {
    let pos = vec2<f32>(
        position.x, position.y
    );
    let clamped_pos = vec2<f32>(particle_pos.x, particle_pos.y);
    return vec4<f32>(pos + clamped_pos, 0.0, 1.0);
}

[[stage(fragment)]]
fn main_fs() -> [[location(0)]] vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
