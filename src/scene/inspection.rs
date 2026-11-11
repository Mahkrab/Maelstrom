use super::ResolvedScene;
use std::fmt::Write;

impl ResolvedScene {
    /// Produces a text representation of the resolved input.
    ///
    /// It retains display metadata for inspection, alongside the ordered
    /// planes, particles and resolved configuration comparison key.
    #[must_use]
    pub fn canonical_inspection(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("scene_schema_version = {}\n", self.schema_version));
        output.push_str(&format!("name = {}\n", quoted(&self.name)));
        output.push_str(&format!("description = {}\n", quoted(&self.description)));
        output.push_str("randomness = \"none\"\n");
        output.push_str(&format!(
            "configuration_key = {}\n",
            quoted(&self.configuration.comparison_key())
        ));
        output.push_str(&format!("plane_count = {}\n", self.planes.len()));
        for (index, plane) in self.planes.iter().enumerate() {
            let normal = plane.normal();
            output.push_str(&format!(
                "plane.{index} = [{:?}, {:?}, {:?}, {:?}]\n",
                normal.x,
                normal.y,
                normal.z,
                plane.offset()
            ));
        }
        output.push_str(&format!("particle_count = {}\n", self.particles.len()));
        for particle in &self.particles {
            output.push_str(&format!(
                "particle.{} = [{:?}, {:?}, {:?}, {:?}, {:?}, {:?}]\n",
                particle.identity,
                particle.position_m.x,
                particle.position_m.y,
                particle.position_m.z,
                particle.velocity_m_s.x,
                particle.velocity_m_s.y,
                particle.velocity_m_s.z,
            ));
        }
        output
    }

    /// Produces the stable key used to compare two resolved initial inputs.
    ///
    /// The key includes the resolved configuration and ordered physical input,
    /// but deliberately excludes names and descriptions.
    #[must_use]
    pub fn comparison_key(&self) -> String {
        let mut key = self.key_prefix();
        self.append_key(&self.particles, &mut key);
        key
    }

    pub(crate) fn comparison_key_for(
        &self,
        particles: &[super::Particle],
    ) -> Result<String, std::collections::TryReserveError> {
        let mut key = self.key_prefix();
        // Each particle needs at most 20 identity digits, six 8-digit binary32
        // tokens and separators. Reserve before formatting to make the large
        // allocation recoverable and avoid one temporary String per particle.
        let extra = particles
            .len()
            .checked_mul(75)
            .and_then(|n| {
                self.planes
                    .len()
                    .checked_mul(36)
                    .and_then(|p| n.checked_add(p))
            })
            .and_then(|n| n.checked_add(17))
            .unwrap_or(usize::MAX);
        key.try_reserve_exact(extra)?;
        self.append_key(particles, &mut key);
        Ok(key)
    }

    fn key_prefix(&self) -> String {
        format!(
            "scene-comparison-v1|randomness=none|{}",
            self.configuration.comparison_key()
        )
    }

    fn append_key(&self, particles: &[super::Particle], key: &mut String) {
        // String formatting itself cannot return fmt::Error; the fallible caller
        // has already reserved sufficient capacity for every token.
        key.push_str("|planes");
        for plane in &self.planes {
            let normal = plane.normal();
            let _ = write!(
                key,
                ":{:08x},{:08x},{:08x},{:08x}",
                normal.x.to_bits(),
                normal.y.to_bits(),
                normal.z.to_bits(),
                plane.offset().to_bits(),
            );
        }
        key.push_str("|particles");
        for particle in particles {
            let _ = write!(
                key,
                ":{}:{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}",
                particle.identity,
                particle.position_m.x.to_bits(),
                particle.position_m.y.to_bits(),
                particle.position_m.z.to_bits(),
                particle.velocity_m_s.x.to_bits(),
                particle.velocity_m_s.y.to_bits(),
                particle.velocity_m_s.z.to_bits(),
            );
        }
    }
}

fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
