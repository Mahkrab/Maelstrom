//! Small binary64 oracle evaluated directly from the PBF equations. It shares
//! no production vector, kernel, neighbour, integration or projection arithmetic.
//! Deliberately incorrect variants establish that fixtures distinguish the
//! ordering and membership rules they are intended to exercise.

use maelstrom::{config::ResolvedConfiguration, scene::Particle};

use super::support::poly6_f64;

type Vector = [f64; 3];

#[derive(Clone, Copy, Default)]
pub struct Variant {
    pub rebuild: bool,
    pub stale_distances: bool,
    pub in_place: bool,
    pub final_projection_only: bool,
}

pub struct Oracle {
    pub positions: Vec<Vector>,
    pub velocities: Vec<Vector>,
    pub stages: Vec<Vec<Vector>>,
}

fn add(a: Vector, b: Vector) -> Vector {
    std::array::from_fn(|axis| a[axis] + b[axis])
}

fn subtract(a: Vector, b: Vector) -> Vector {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn scale(a: Vector, factor: f64) -> Vector {
    a.map(|value| value * factor)
}

fn dot(a: Vector, b: Vector) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

fn distance(a: Vector, b: Vector) -> f64 {
    let delta = subtract(a, b);
    dot(delta, delta).sqrt()
}

fn neighbours(positions: &[Vector], h: f64) -> Vec<Vec<usize>> {
    positions
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            positions
                .iter()
                .enumerate()
                .filter_map(|(j, &q)| (i != j && distance(p, q) < h).then_some(j))
                .collect()
        })
        .collect()
}

fn gradient(a: Vector, b: Vector, h: f64) -> Vector {
    let r = distance(a, b);
    if r == 0.0 || r >= h {
        [0.0; 3]
    } else {
        scale(
            subtract(a, b),
            -45.0 * (h - r).powi(2) / (std::f64::consts::PI * h.powi(6) * r),
        )
    }
}

struct Coefficients {
    h: f64,
    rest: f64,
    epsilon: f64,
    pressure: f64,
    reference: f64,
    exponent: f64,
}

fn corrections(
    positions: &[Vector],
    rows: &[Vec<usize>],
    coefficients: &Coefficients,
) -> Vec<Vector> {
    let Coefficients {
        h,
        rest,
        epsilon,
        pressure,
        reference,
        exponent,
    } = *coefficients;
    let mut multipliers = Vec::new();
    for (i, &position) in positions.iter().enumerate() {
        let density: f64 = (0..positions.len())
            .filter(|&j| i == j || rows[i].contains(&j))
            .map(|j| poly6_f64(distance(position, positions[j]), h))
            .sum();
        let owner = scale(
            rows[i].iter().fold([0.0; 3], |sum, &j| {
                add(sum, gradient(position, positions[j], h))
            }),
            1.0 / rest,
        );
        let squared_directions: f64 = (0..positions.len())
            .map(|j| {
                let direction = if i == j {
                    owner
                } else if rows[i].contains(&j) {
                    scale(gradient(position, positions[j], h), -1.0 / rest)
                } else {
                    [0.0; 3]
                };
                dot(direction, direction)
            })
            .sum();
        multipliers.push(-(density / rest - 1.0) / (squared_directions + epsilon));
    }
    positions
        .iter()
        .enumerate()
        .map(|(i, &position)| {
            scale(
                rows[i].iter().fold([0.0; 3], |sum, &j| {
                    let r = distance(position, positions[j]);
                    let scorr =
                        -pressure * (poly6_f64(r, h) / poly6_f64(reference, h)).powf(exponent);
                    add(
                        sum,
                        scale(
                            gradient(position, positions[j], h),
                            multipliers[i] + multipliers[j] + scorr,
                        ),
                    )
                }),
                1.0 / rest,
            )
        })
        .collect()
}

pub fn step(
    particles: &[Particle],
    config: &ResolvedConfiguration,
    planes: &[(Vector, f64)],
    iterations: usize,
    variant: Variant,
) -> Oracle {
    let model = &config.numerical;
    let dt = f64::from(model.fixed_timestep_s);
    let spacing = f64::from(model.particle_spacing_m);
    let h = spacing * f64::from(model.support_radius_ratio);
    // Re-derive rest density from lattice offsets, never from production's sum.
    let extent = model.support_radius_ratio.ceil() as i32;
    let mut rest = 0.0;
    for x in -extent..=extent {
        for y in -extent..=extent {
            for z in -extent..=extent {
                let r = spacing * f64::from(x * x + y * y + z * z).sqrt();
                rest += poly6_f64(r, h);
            }
        }
    }
    let epsilon = f64::from(model.relaxation_scale) / h.powi(2);
    let pressure = if model.artificial_pressure_enabled {
        f64::from(model.artificial_pressure_strength_scale) * h.powi(2)
    } else {
        0.0
    };
    let reference = f64::from(model.artificial_pressure_reference_separation_ratio) * h;
    let exponent = f64::from(model.artificial_pressure_exponent);
    let coefficients = Coefficients {
        h,
        rest,
        epsilon,
        pressure,
        reference,
        exponent,
    };
    let convert = |p: maelstrom::math::Vec3| [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
    let acceleration = convert(model.external_acceleration_m_s2);
    let initial: Vec<_> = particles.iter().map(|p| convert(p.position_m)).collect();
    let mut positions: Vec<_> = particles
        .iter()
        .map(|p| {
            add(
                convert(p.position_m),
                scale(add(convert(p.velocity_m_s), scale(acceleration, dt)), dt),
            )
        })
        .collect();
    let prediction = positions.clone();
    let mut rows = neighbours(&prediction, h);
    let mut stages = vec![positions.clone()];
    for iteration in 0..iterations {
        if variant.rebuild {
            rows = neighbours(&positions, h);
        }
        let input = if variant.stale_distances {
            &prediction
        } else {
            &positions
        };
        let delta: Vec<_> = corrections(input, &rows, &coefficients)
            .into_iter()
            .map(|correction| scale(correction, f64::from(model.correction_scale)))
            .collect();
        for i in 0..positions.len() {
            let correction = if variant.in_place {
                scale(
                    corrections(&positions, &rows, &coefficients)[i],
                    f64::from(model.correction_scale),
                )
            } else {
                delta[i]
            };
            positions[i] = add(positions[i], correction);
            if !variant.final_projection_only || iteration + 1 == iterations {
                for &(normal, offset) in planes {
                    let constraint = dot(normal, positions[i]) - offset;
                    if constraint < 0.0 {
                        positions[i] = subtract(positions[i], scale(normal, constraint));
                    }
                }
            }
        }
        stages.push(positions.clone());
    }
    let velocities = positions
        .iter()
        .zip(initial)
        .map(|(&p, old)| scale(subtract(p, old), 1.0 / dt))
        .collect();
    Oracle {
        positions,
        velocities,
        stages,
    }
}

pub fn maximum_difference(a: &[Vector], b: &[Vector]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}
