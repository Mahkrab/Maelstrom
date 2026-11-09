// View-relative metres are scaled uniformly by the eye-target distance on upload.
struct Camera {
    projection: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    light_radius: vec4<f32>,
    colour_low: vec4<f32>,
    colour_high: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) @interpolate(flat) colour_fraction: f32, };
fn project(p: vec3<f32>) -> vec4<f32> {
    return vec4(p.x * camera.projection.x, p.y * camera.projection.y,
        p.z * camera.projection.z + camera.projection.w, p.z);
}
@vertex fn sphere(@location(0) normal: vec3<f32>, @location(1) centre: vec3<f32>, @location(2) colour_fraction: f32) -> Vertex {
    let view_normal = vec3(dot(normal, camera.right.xyz),dot(normal, camera.up.xyz),dot(normal,camera.forward.xyz));
    var result: Vertex;
    result.position = project(centre + view_normal * camera.light_radius.w);
    result.normal = normal;
    result.colour_fraction = colour_fraction;
    return result;
}
@vertex fn line(@location(0) point: vec3<f32>) -> Vertex {
    var result: Vertex; result.position = project(point); result.normal = vec3(0.0); result.colour_fraction = 0.0; return result;
}
@fragment fn particle_colour(vertex: Vertex) -> @location(0) vec4<f32> {
    let lighting = 0.28 + 0.72 * max(dot(normalize(vertex.normal), normalize(vec3(-0.4,0.8,0.6))),0.0);
    let base = mix(camera.colour_low.xyz, camera.colour_high.xyz, vertex.colour_fraction);
    return vec4(base * lighting,1.0);
}
@fragment fn line_colour() -> @location(0) vec4<f32> { return vec4(0.323143,0.391572,0.434154,1.0); }

// Tangent bounds cover the perspective silhouette, including off-axis spheres.
// Near/eye-plane intersections use the indexed mesh before reaching this pass.
struct ImpostorVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(linear) ray: vec2<f32>,
    @location(1) @interpolate(flat) centre: vec3<f32>,
    @location(2) @interpolate(flat) colour_fraction: f32,
};
@vertex fn impostor(@location(0) corner: vec3<f32>, @location(1) centre: vec3<f32>,
                    @location(2) colour_fraction: f32) -> ImpostorVertex {
    let radius = camera.light_radius.w;
    let denominator = centre.z * centre.z - radius * radius;
    let middle = centre.xy * centre.z / denominator;
    let extent = radius * sqrt(centre.xy * centre.xy + vec2(denominator)) / denominator;
    let slope = middle + corner.xy * (extent * 1.000002 + vec2(0.0000001));
    var result: ImpostorVertex;
    result.position = vec4(slope * camera.projection.xy, 0.0, 1.0);
    result.ray = slope;
    result.centre = centre;
    result.colour_fraction = colour_fraction;
    return result;
}
struct ImpostorFragment {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};
@fragment fn impostor_colour(vertex: ImpostorVertex) -> ImpostorFragment {
    let radius = camera.light_radius.w;
    let ray = vec3(vertex.ray, 1.0);
    let ray_squared = dot(ray, ray);
    let middle = dot(ray, vertex.centre) / ray_squared;
    // Closest approach avoids subtracting two almost equal squared distances.
    let closest = vertex.centre - ray * middle;
    let height_squared = radius * radius - dot(closest, closest);
    if height_squared < 0.0 { discard; }
    let half_chord = sqrt(max(height_squared, 0.0) / ray_squared);
    let z = middle - half_chord;
    if z <= 0.0 { discard; }
    let depth = (z * camera.projection.z + camera.projection.w) / z;
    if depth < 0.0 || depth > 1.0 { discard; }
    let view_normal = (-closest - ray * half_chord) / radius;
    let normal = view_normal.x * camera.right.xyz + view_normal.y * camera.up.xyz
        + view_normal.z * camera.forward.xyz;
    let lighting = 0.28 + 0.72 * max(dot(normalize(normal), normalize(vec3(-0.4,0.8,0.6))),0.0);
    let base = mix(camera.colour_low.xyz, camera.colour_high.xyz, vertex.colour_fraction);
    var result: ImpostorFragment;
    result.colour = vec4(base * lighting, 1.0);
    result.depth = depth;
    return result;
}
