use super::ResolvedScene;

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
        let mut key = format!(
            "scene-comparison-v1|randomness=none|{}",
            self.configuration.comparison_key()
        );
        key.push_str("|planes");
        for plane in &self.planes {
            let normal = plane.normal();
            key.push_str(&format!(
                ":{:08x},{:08x},{:08x},{:08x}",
                normal.x.to_bits(),
                normal.y.to_bits(),
                normal.z.to_bits(),
                plane.offset().to_bits(),
            ));
        }
        key.push_str("|particles");
        for particle in &self.particles {
            key.push_str(&format!(
                ":{}:{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}",
                particle.identity,
                particle.position_m.x.to_bits(),
                particle.position_m.y.to_bits(),
                particle.position_m.z.to_bits(),
                particle.velocity_m_s.x.to_bits(),
                particle.velocity_m_s.y.to_bits(),
                particle.velocity_m_s.z.to_bits(),
            ));
        }
        key
    }
}

fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
