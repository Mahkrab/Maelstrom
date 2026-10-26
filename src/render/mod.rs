//! Drawing consumes accepted particles and frozen initial geometry. Graphics
//! resources have no dependency back into solver acceptance.

pub(crate) use crate::presentation::camera;
pub use crate::presentation::{Camera, RenderError, Settings};
mod geometry;
mod gpu;
pub use geometry::BoundaryLines;
pub use gpu::{DrawScene, Graphics, GraphicsSelection};
