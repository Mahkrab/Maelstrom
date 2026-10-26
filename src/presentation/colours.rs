//! Colour maps consume accepted velocities and physical particle mass. Their
//! fixed ranges are shared by windowed drawing and offline image production.

use super::{ParticleColours, RenderError};
use crate::scene::Particle;

/// Validated colour parameters prepared once before packing a frame.
pub struct ColourMap {
    low: [f32; 3],
    high: [f32; 3],
    energy: Option<(f64, f64, f64)>,
}

impl ParticleColours {
    /// Prepares linear RGB endpoints and, when required, a kinetic-energy interval.
    ///
    /// # Errors
    /// Rejects malformed colours, invalid energy bounds and non-positive/non-finite mass.
    pub fn prepare(&self, mass_kg: f64) -> Result<ColourMap, RenderError> {
        match self {
            Self::Fixed { colour } => {
                let rgb = if colour.eq_ignore_ascii_case("#2b9fd9") {
                    // Retain the original renderer's rounded constants for its
                    // default colour, including its existing decoded pixels.
                    [0.0241576, 0.346704, 0.693872]
                } else {
                    if colour.len() != 7
                        || !colour.starts_with('#')
                        || !colour.as_bytes()[1..].iter().all(u8::is_ascii_hexdigit)
                    {
                        return Err(RenderError("particles.colour requires #RRGGBB".into()));
                    }
                    let mut rgb = [0.0; 3];
                    for (axis, channel) in rgb.iter_mut().enumerate() {
                        let value = u8::from_str_radix(&colour[1 + 2 * axis..3 + 2 * axis], 16)
                            .map_err(|_| RenderError("particles.colour requires #RRGGBB".into()))?;
                        let value = f64::from(value) / 255.0;
                        *channel = if value <= 0.04045 {
                            value / 12.92
                        } else {
                            ((value + 0.055) / 1.055).powf(2.4)
                        } as f32;
                    }
                    rgb
                };
                Ok(ColourMap {
                    low: rgb,
                    high: rgb,
                    energy: None,
                })
            }
            Self::KineticEnergy {
                min_energy_j: min,
                max_energy_j: max,
            } => {
                if !min.is_finite()
                    || !max.is_finite()
                    || *min < 0.0
                    || max <= min
                    || !mass_kg.is_finite()
                    || mass_kg <= 0.0
                {
                    return Err(RenderError("energy colouring requires finite 0 <= min_energy_j < max_energy_j and positive finite particle mass".into()));
                }
                Ok(ColourMap {
                    low: [0.0, 0.0, 1.0],
                    high: [1.0, 0.0, 0.0],
                    energy: Some((*min, *max, mass_kg)),
                })
            }
        }
    }
}

impl ColourMap {
    /// Linear RGB endpoints, before sphere lighting and sRGB encoding.
    #[must_use]
    pub fn endpoints(&self) -> ([f32; 3], [f32; 3]) {
        (self.low, self.high)
    }

    /// Computes a clamped colour fraction from an accepted particle's velocity.
    ///
    /// # Errors
    /// Rejects non-finite velocity or energy when energy colouring is selected.
    pub fn fraction(&self, particle: &Particle) -> Result<f32, RenderError> {
        let Some((min, max, mass)) = self.energy else {
            return Ok(0.0);
        };
        let velocity = particle.velocity_m_s;
        let [x, y, z] = [
            f64::from(velocity.x),
            f64::from(velocity.y),
            f64::from(velocity.z),
        ];
        let energy = 0.5 * mass * (x * x + y * y + z * z);
        if !energy.is_finite() {
            return Err(RenderError(
                "energy colouring requires finite accepted velocities and kinetic energy".into(),
            ));
        }
        // Clamp first: dividing a large energy by a subnormal range could overflow.
        Ok(if energy <= min {
            0.0
        } else if energy >= max {
            1.0
        } else {
            ((energy - min) / (max - min)) as f32
        })
    }
}
