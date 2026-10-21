//! Completed image sequences are read-only presentation inputs. Loading them
//! never resolves physical inputs, reconstructs particles or acquires a solver.

#[cfg(any(feature = "viewer", test))]
mod clock;
#[cfg(feature = "viewer")]
mod image;
mod sequence;
#[cfg(test)]
mod tests;
#[cfg(feature = "viewer")]
mod window;

pub use sequence::Sequence;
#[cfg(feature = "viewer")]
pub use window::run;
