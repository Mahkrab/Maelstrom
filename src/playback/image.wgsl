// A saved sRGB image is sampled without lighting, camera transforms or geometry.
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
    let points = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3));
    let p = points[index];
    return Vertex(vec4<f32>(p, 0, 1), vec2<f32>((p.x+1)*0.5, (1-p.y)*0.5));
}
@fragment fn fragment(input: Vertex) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, input.uv);
}
