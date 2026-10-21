struct Vertex { @builtin(position) position: vec4<f32>, @location(0) colour: vec3<f32>, };
@vertex fn overlay(@location(0) position: vec2<f32>, @location(1) colour: vec3<f32>) -> Vertex {
    var result: Vertex; result.position=vec4(position,0.0,1.0); result.colour=colour; return result;
}
@fragment fn colour(vertex: Vertex) -> @location(0) vec4<f32> { return vec4(vertex.colour,1.0); }
