//! Presentation documents can be inspected without a graphics build or device.
//! The renderer receives validated drawing settings, never numerical overrides.

mod bounds;
pub(crate) mod camera;
mod colours;
#[cfg(feature = "viewer")]
mod save;
mod settings;
pub use bounds::domain_bounds;
pub use camera::Camera;
pub use colours::ColourMap;
#[cfg(feature = "viewer")]
pub use save::{CameraSaveError, CameraSaveTarget};
use serde::{Deserialize, Serialize};
pub use settings::{RenderError, Settings};

/// Camera selection before a viewer or exporter fits its initial bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraMode {
    /// Fit once to the initial scene bounds.
    Fit,
    /// Use the selected fixed camera.
    Fixed,
}

/// Presentation-only particle colour request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "colour_mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParticleColours {
    /// One sRGB colour for every particle.
    Fixed { colour: String },
    /// A fixed kinetic-energy interval in joules.
    KineticEnergy {
        min_energy_j: f64,
        max_energy_j: f64,
    },
}

impl Default for ParticleColours {
    fn default() -> Self {
        Self::Fixed {
            colour: "#2b9fd9".into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema_version: u32,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
    step_stride: Option<u64>,
    camera: Option<toml::Value>,
    camera_mode: Option<CameraMode>,
    particles: Option<ColourDocument>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColourDocument {
    colour_mode: Option<String>,
    colour: Option<String>,
    min_energy_j: Option<f64>,
    max_energy_j: Option<f64>,
}

/// Validated presentation selection, including the input version and field origins.
#[derive(Clone, Debug)]
pub struct Presentation {
    /// Input document version, independently of the legacy drawing record format.
    pub schema_version: u32,
    /// Dimensions, image schedule and the selected fixed camera or fallback.
    pub settings: Settings,
    /// Selected fit/fixed policy, before explicit flags.
    pub camera_mode: CameraMode,
    /// Fields explicitly supplied by the selected document.
    pub supplied_fields: Vec<String>,
}

impl Default for Presentation {
    fn default() -> Self {
        Self {
            schema_version: 2,
            settings: Settings::default(),
            camera_mode: CameraMode::Fit,
            supplied_fields: Vec::new(),
        }
    }
}

impl Presentation {
    /// Parses either supported document version without initialising graphics.
    ///
    /// # Errors
    /// Rejects unknown fields, invalid domains, unsupported versions and contradictory modes.
    pub fn parse(source: &str) -> Result<Self, RenderError> {
        let doc: Document =
            toml::from_str(source).map_err(|e| RenderError(format!("render settings: {e}")))?;
        if !matches!(doc.schema_version, 1 | 2) {
            return Err(RenderError(format!(
                "unsupported render schema_version {}",
                doc.schema_version
            )));
        }
        if doc.schema_version == 1 && (doc.camera_mode.is_some() || doc.particles.is_some()) {
            return Err(RenderError(
                "camera_mode and particles require render schema 2".into(),
            ));
        }
        let mut supplied_fields = Vec::new();
        for (field, present) in [
            ("width", doc.width.is_some()),
            ("height", doc.height.is_some()),
            ("format", doc.format.is_some()),
            ("step_stride", doc.step_stride.is_some()),
            ("camera", doc.camera.is_some()),
            ("camera_mode", doc.camera_mode.is_some()),
            ("particles", doc.particles.is_some()),
        ] {
            if present {
                supplied_fields.push(field.into());
            }
        }
        let camera_mode =
            doc.camera_mode
                .unwrap_or(if doc.schema_version == 1 || doc.camera.is_some() {
                    CameraMode::Fixed
                } else {
                    CameraMode::Fit
                });
        if doc.schema_version == 2
            && ((camera_mode == CameraMode::Fixed && doc.camera.is_none())
                || (doc.camera_mode == Some(CameraMode::Fit) && doc.camera.is_some()))
        {
            return Err(RenderError(
                "fixed camera requires [camera]; explicit fit must not include [camera]".into(),
            ));
        }
        let camera: Option<Camera> = doc
            .camera
            .map(|value| {
                if let Some(fields) = value.as_table() {
                    supplied_fields.extend(fields.keys().map(|field| format!("camera.{field}")));
                }
                value
                    .try_into()
                    .map_err(|e| RenderError(format!("render camera: {e}")))
            })
            .transpose()?;
        let defaults = Settings::default();
        let mut settings = Settings {
            schema_version: 1,
            width: doc.width.unwrap_or(defaults.width),
            height: doc.height.unwrap_or(defaults.height),
            format: doc.format.unwrap_or(defaults.format),
            step_stride: doc.step_stride.unwrap_or(defaults.step_stride),
            camera: camera.unwrap_or(defaults.camera),
            particles: ParticleColours::default(),
        };
        settings.validate()?;
        let colour = doc.particles.unwrap_or_default();
        for (field, present) in [
            ("colour_mode", colour.colour_mode.is_some()),
            ("colour", colour.colour.is_some()),
            ("min_energy_j", colour.min_energy_j.is_some()),
            ("max_energy_j", colour.max_energy_j.is_some()),
        ] {
            if present {
                supplied_fields.push(format!("particles.{field}"));
            }
        }
        let particles = match colour.colour_mode.as_deref().unwrap_or("fixed") {
            "fixed" if colour.min_energy_j.is_none() && colour.max_energy_j.is_none() => {
                let value = colour.colour.unwrap_or_else(|| "#2b9fd9".into());
                if value.len()!=7 || !value.starts_with('#') || !value.as_bytes()[1..].iter().all(u8::is_ascii_hexdigit) {
                    return Err(RenderError("particles.colour requires #RRGGBB".into()));
                }
                ParticleColours::Fixed { colour: value.to_ascii_lowercase() }
            }
            "kinetic_energy" if colour.colour.is_none() => {
                let (Some(min),Some(max)) = (colour.min_energy_j,colour.max_energy_j) else {
                    return Err(RenderError("kinetic_energy requires min_energy_j and max_energy_j".into()));
                };
                if !min.is_finite() || !max.is_finite() || min<0.0 || max<=min {
                    return Err(RenderError("particles energy bounds require finite 0 <= min_energy_j < max_energy_j".into()));
                }
                ParticleColours::KineticEnergy { min_energy_j:min,max_energy_j:max }
            }
            _ => return Err(RenderError("particles.colour_mode must be fixed without energy bounds or kinetic_energy without colour".into())),
        };
        settings.particles = particles.clone();
        if particles != ParticleColours::default() {
            settings.schema_version = 2;
        }
        settings.validate()?;
        Ok(Self {
            schema_version: doc.schema_version,
            settings,
            camera_mode,
            supplied_fields,
        })
    }

    /// Whether the current drawing implementation supports this colour selection.
    #[must_use]
    pub fn drawing_supported(&self) -> bool {
        self.settings.validate().is_ok()
    }

    /// Encodes a portable new document, retaining the resolved fixed camera when selected.
    ///
    /// # Errors
    /// Returns a serialisation error if the document cannot be encoded as TOML.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        #[derive(Serialize)]
        struct Saved<'a> {
            schema_version: u32,
            width: u32,
            height: u32,
            format: &'a str,
            step_stride: u64,
            camera_mode: CameraMode,
            #[serde(skip_serializing_if = "Option::is_none")]
            camera: Option<&'a Camera>,
            particles: &'a ParticleColours,
        }
        toml::to_string_pretty(&Saved {
            schema_version: 2,
            width: self.settings.width,
            height: self.settings.height,
            format: &self.settings.format,
            step_stride: self.settings.step_stride,
            camera_mode: self.camera_mode,
            camera: (self.camera_mode == CameraMode::Fixed).then_some(&self.settings.camera),
            particles: &self.settings.particles,
        })
    }
}

fn serialize_settings<S: serde::Serializer>(
    settings: &Settings,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Values<'a> {
        width: u32,
        height: u32,
        format: &'a str,
        step_stride: u64,
        camera: &'a Camera,
    }
    Values {
        width: settings.width,
        height: settings.height,
        format: &settings.format,
        step_stride: settings.step_stride,
        camera: &settings.camera,
    }
    .serialize(serializer)
}

impl Serialize for Presentation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Effective<'a> {
            schema_version: u32,
            #[serde(serialize_with = "serialize_settings")]
            settings: &'a Settings,
            camera_mode: CameraMode,
            particles: &'a ParticleColours,
        }
        Effective {
            schema_version: self.schema_version,
            settings: &self.settings,
            camera_mode: self.camera_mode,
            particles: &self.settings.particles,
        }
        .serialize(serializer)
    }
}
