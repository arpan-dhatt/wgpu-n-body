struct CameraUniform {
    view_proj: mat4x4<f32>;
};

[[group(0), binding(0)]] var<uniform> camera: CameraUniform;

[[stage(vertex)]]
fn main_vs(
    [[location(0)]] particle_pos: vec3<f32>,
    [[location(1)]] particle_vel: vec3<f32>,
    [[location(3)]] position: vec2<f32>,
) -> [[builtin(position)]] vec4<f32> {
    let v_pos = vec4<f32>(
        position.x, position.y, 0.0, 0.0
    );
    return camera.view_proj * vec4<f32>(particle_pos, 1.0) + v_pos;
}

[[stage(fragment)]]
fn main_fs() -> [[location(0)]] vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 0.25);
}
