//! Accepted simulation state exposed through immutable snapshots.
//!
//! Identity validation preserves the index order used by serial reductions
//! while callers inspect or clone a completed step.

use crate::{scene::Particle, validation};

use super::StepError;

/// Accepted particles at a completed step boundary.
///
/// Particles are held in ascending identity order, with positions in metres and
/// velocities in metres per second. Cloning retains an independent checkpoint;
/// neither borrowed observations nor owned checkpoints permit state mutation.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub(super) completed_steps: u64,
    pub(super) particles: Vec<Particle>,
}

impl Snapshot {
    /// Number of fully accepted steps since construction; initial state is zero.
    #[must_use]
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Borrows all particles in ascending identity order.
    ///
    /// ```compile_fail
    /// use maelstrom::simulation::Snapshot;
    /// fn change_state(snapshot: &Snapshot) {
    ///     snapshot.particles()[0].position_m.x = 0.0;
    /// }
    /// ```
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }
}

pub(super) fn validate_particles(particles: &[Particle]) -> Result<(), StepError> {
    let mut previous = None;
    for particle in particles {
        if let Some(previous) = previous
            && previous >= particle.identity
        {
            return Err(StepError::IdentityOrder {
                previous,
                next: particle.identity,
            });
        }
        validation::finite_vector("particle position", particle.position_m)?;
        validation::finite_vector("particle velocity", particle.velocity_m_s)?;
        previous = Some(particle.identity);
    }
    Ok(())
}
