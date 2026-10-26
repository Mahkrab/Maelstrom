//! Initial grouped geometry uses fixed spatial cells to avoid all-pairs scans.
//! Lookup order and bucket insertion are explicit; hash-map iteration is never used.

use super::{PARTICLE_SEPARATION_RELATIVE_TOLERANCE, Particle, SceneError};
use crate::config::ConfigError;
use std::{collections::HashMap, ops::Range};

pub(super) fn validate_separation(
    particles: &[Particle],
    diameter: f32,
    allow_coincident: bool,
    lattice_ranges: &[Range<usize>],
) -> Result<(), SceneError> {
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::with_capacity(particles.len());
    let tolerance = diameter * PARTICLE_SEPARATION_RELATIVE_TOLERANCE;
    let mut lattice = 0;
    for (index, particle) in particles.iter().enumerate() {
        while lattice_ranges
            .get(lattice)
            .is_some_and(|range| range.end <= index)
        {
            lattice += 1;
        }
        let lattice_start = lattice_ranges
            .get(lattice)
            .filter(|range| range.start <= index)
            .map(|range| range.start);
        let position = particle.position_m;
        let values = [position.x, position.y, position.z];
        let mut cell = [0; 3];
        for axis in 0..3 {
            let coordinate = (f64::from(values[axis]) / f64::from(diameter)).floor();
            if !coordinate.is_finite()
                || coordinate < i64::MIN as f64
                || coordinate >= i64::MAX as f64
            {
                return Err(ConfigError::InvalidField {field:"particles.position_m",reason:"grouped coordinates divided by spacing must fit signed 64-bit spatial cells".into()}.into());
            }
            cell[axis] = coordinate as i64;
        }
        let mut overlap: Option<usize> = None;
        let mut coincident = false;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let (Some(x), Some(y), Some(z)) = (
                        cell[0].checked_add(x),
                        cell[1].checked_add(y),
                        cell[2].checked_add(z),
                    ) else {
                        continue;
                    };
                    if let Some(bucket) = cells.get(&[x, y, z]) {
                        for &previous in bucket {
                            // Construction already checked each lattice axis. Rechecking its
                            // nominal spacing against rounded world coordinates would reject
                            // valid translated decimal grids. Other sources still need checks.
                            if lattice_start.is_some_and(|start| previous >= start) {
                                continue;
                            }
                            let distance = (particle.position_m - particles[previous].position_m)
                                .checked_length("initial particle separation")?;
                            if particle.position_m == particles[previous].position_m
                                && allow_coincident
                            {
                                coincident = true;
                                continue;
                            }
                            if distance + tolerance < diameter {
                                overlap = Some(overlap.map_or(previous, |old| old.min(previous)));
                            }
                        }
                    }
                }
            }
        }
        if let Some(previous) = overlap {
            return Err(SceneError::OverlappingParticles {
                first: particles[previous].identity,
                second: particle.identity,
            });
        }
        // One representative suffices for exactly coincident accepted positions:
        // every later distance is identical, including the chosen earliest identity.
        if !coincident {
            cells.entry(cell).or_default().push(index);
        }
    }
    Ok(())
}
