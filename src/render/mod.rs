//! Drawing consumes accepted particles and frozen initial geometry. Graphics
//! resources and camera settings have no dependency back into solver acceptance.

mod camera;
mod geometry;
mod gpu;
pub use camera::Camera;
pub use geometry::BoundaryLines;
pub use gpu::{DrawScene, Graphics, GraphicsSelection};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A graphics, geometry or presentation-input failure.
#[derive(Debug, Clone)]
pub struct RenderError(pub String);
impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for RenderError {}

/// Strict presentation-only schema. Physical configuration precedence is separate.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Required schema version, currently 1.
    pub schema_version: u32,
    /// Drawable width, 1..=8192.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Drawable height, 1..=8192.
    #[serde(default = "default_height")]
    pub height: u32,
    /// Image format for offline output; only png.
    #[serde(default = "default_format")]
    pub format: String,
    /// Accepted-step stride for offline output; positive.
    #[serde(default = "default_stride")]
    pub step_stride: u64,
    /// Camera fields merge with built-in defaults.
    #[serde(default)]
    pub camera: Camera,
}
fn default_width() -> u32 {
    1280
}
fn default_height() -> u32 {
    720
}
fn default_format() -> String {
    "png".into()
}
fn default_stride() -> u64 {
    1
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            schema_version: 1,
            width: 1280,
            height: 720,
            format: default_format(),
            step_stride: 1,
            camera: Camera::default(),
        }
    }
}
impl Settings {
    /// Parses a selected render document, rejecting unknown fields and versions.
    pub fn parse(source: &str) -> Result<Self, RenderError> {
        let settings: Self =
            toml::from_str(source).map_err(|e| RenderError(format!("render settings: {e}")))?;
        settings.validate()?;
        Ok(settings)
    }
    /// Validates settings before window or output creation.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.schema_version != 1
            || !(1..=8192).contains(&self.width)
            || !(1..=8192).contains(&self.height)
            || self.format != "png"
            || self.step_stride == 0
        {
            return Err(RenderError("render settings require schema 1, dimensions 1..=8192, png and positive step_stride".into()));
        }
        let _bytes = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| RenderError("RGBA dimensions overflow".into()))?;
        self.camera
            .validate(f64::from(self.width) / f64::from(self.height))
    }
}
