//! Optional CUDA resources support the device solver and an independent layout probe.
//! Successful calls complete before returning. Uncertain device work invalidates
//! its resource and cannot publish output.

use std::{error::Error, fmt};

use serde::Serialize;

#[cfg(feature = "cuda")]
mod native;
#[cfg(feature = "cuda")]
pub(crate) mod solver;

/// Explicit transfer layout: identity is untouched; the probe transforms three
/// binary32 coordinates and a marker. Size is 24 bytes, alignment is eight.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Probe {
    pub identity: u64,
    pub position: [f32; 3],
    pub marker: u32,
}

const _: () = {
    assert!(size_of::<Probe>() == 24 && align_of::<Probe>() == 8);
    assert!(std::mem::offset_of!(Probe, position) == 8);
    assert!(std::mem::offset_of!(Probe, marker) == 20);
};

/// The operation that failed; launch submission and device completion are distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaStage {
    DriverLoad,
    Initialise,
    DeviceSelection,
    Context,
    Module,
    Stream,
    Allocation,
    Upload,
    Launch,
    Completion,
    Download,
    Release,
    ContextBinding,
    Close,
}

/// CUDA failures never select a CPU implementation. Only host validation and
/// allocation exhaustion permit retry on the same resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    FeatureDisabled,
    InvalidDevice {
        ordinal: u32,
    },
    Capacity,
    HostAllocation,
    Poisoned,
    Driver {
        stage: CudaStage,
        code: i32,
        invalidated: bool,
    },
    Verification {
        index: usize,
    },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureDisabled => write!(
                f,
                "CUDA support is disabled; rebuild with --features cuda (see docs/cuda.md)"
            ),
            Self::InvalidDevice { ordinal } => write!(
                f,
                "CUDA device ordinal {ordinal} is unavailable; select a visible device with --device"
            ),
            Self::Capacity => write!(
                f,
                "CUDA count or byte size exceeds the supported u32/isize capacity"
            ),
            Self::HostAllocation => write!(f, "could not allocate CUDA probe host output"),
            Self::Poisoned => write!(
                f,
                "CUDA resource is invalid after an earlier failure; close it and construct a new resource"
            ),
            Self::Driver {
                stage,
                code,
                invalidated,
            } => {
                write!(
                    f,
                    "CUDA {stage:?} failed (driver code {code}; resource invalidated: {invalidated})"
                )?;
                if *stage == CudaStage::DriverLoad {
                    write!(
                        f,
                        "; a CUDA 13-compatible system libcuda.so.1 is required (see docs/cuda.md)"
                    )?;
                } else {
                    let detail = match code {
                        2 => Some("device memory allocation exhausted"),
                        100 => Some("no CUDA device is visible"),
                        209 => Some("device does not support the compiled CUDA target"),
                        222 => Some(
                            "driver cannot load this toolkit's PTX; use a compatible driver/toolkit",
                        ),
                        _ => None,
                    };
                    if let Some(detail) = detail {
                        write!(f, "; {detail}")?;
                    }
                }
                Ok(())
            }
            Self::Verification { index } => write!(
                f,
                "CUDA layout probe disagreed with the independent expected value at index {index}"
            ),
        }
    }
}

impl Error for CudaError {}

/// Observed device properties; `target` is the embedded PTX virtual architecture.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    pub ordinal: u32,
    pub name: String,
    pub memory_bytes: u64,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub driver_version: i32,
    pub target: String,
}

/// A private context, module and non-blocking stream on one explicitly selected
/// device. It stays on its creating thread and never resets a shared context.
/// Dropping attempts cleanup; [`Self::close`] additionally reports teardown errors.
pub struct CudaContext {
    _private: (),
    #[cfg(feature = "cuda")]
    inner: native::Context,
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaContext")
            .field("device", self.device())
            .finish_non_exhaustive()
    }
}

impl CudaContext {
    /// Open a device by its process-visible ordinal. An incompatible target,
    /// missing driver/device or unavailable feature returns an explicit error.
    pub fn new(ordinal: u32) -> Result<Self, CudaError> {
        i32::try_from(ordinal).map_err(|_| CudaError::InvalidDevice { ordinal })?;
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                _private: (),
                inner: native::Context::new(ordinal)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::FeatureDisabled)
        }
    }

    /// The metadata captured when this context was constructed.
    pub fn device(&self) -> &DeviceInfo {
        #[cfg(feature = "cuda")]
        {
            &self.inner.info
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!("construction requires the CUDA feature")
        }
    }

    /// Copy records to the device, transform them, wait, then return independent
    /// output. Coordinates become `(x + 1, y * -2, z - 0.5)` and markers are XORed
    /// with `0xa5a55a5a`; identities stay exact. Empty input launches no work.
    /// Input is read-only. No partial result escapes a failed operation.
    pub fn roundtrip(&mut self, input: &[Probe]) -> Result<Vec<Probe>, CudaError> {
        checked_bytes(input.len())?;
        #[cfg(feature = "cuda")]
        {
            self.inner.roundtrip(input)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::FeatureDisabled)
        }
    }

    /// Consume the context and report any driver teardown failure. The resource
    /// cannot be retried after close, including when destruction fails.
    pub fn close(self) -> Result<(), CudaError> {
        #[cfg(feature = "cuda")]
        {
            self.inner.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::FeatureDisabled)
        }
    }
}

fn checked_bytes(count: usize) -> Result<usize, CudaError> {
    if count > u32::MAX as usize {
        return Err(CudaError::Capacity);
    }
    count
        .checked_mul(size_of::<Probe>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(CudaError::Capacity)
}

/// Evidence from repeated independent resource lifetimes at the public boundary.
#[derive(Debug, Serialize)]
pub struct ProbeReport {
    pub device: DeviceInfo,
    pub repetitions: u32,
    pub records_per_repetition: usize,
    pub verified: bool,
}

/// Verify explicit expected values through the same library boundary as callers.
/// Each repetition creates and closes its own resource. The 257 records exercise
/// a partially occupied final block with identities larger than binary32 precision.
pub fn verify_roundtrip(
    ordinal: u32,
    repetitions: std::num::NonZeroU32,
) -> Result<ProbeReport, CudaError> {
    let input = vec![
        Probe {
            identity: 0x1234_5678_9abc_def0,
            position: [2.0, -3.0, 4.5],
            marker: 0x1234_5678
        };
        257
    ];
    let mut device = None;
    for _ in 0..repetitions.get() {
        let mut context = CudaContext::new(ordinal)?;
        let output = context.roundtrip(&input)?;
        let mismatch = output.iter().position(|value| {
            value.identity != 0x1234_5678_9abc_def0
                || value.position.map(f32::to_bits) != [3.0_f32, 6.0, 4.0].map(f32::to_bits)
                || value.marker != 0xb791_0c22
        });
        device = Some(context.device().clone());
        context.close()?;
        if let Some(index) = mismatch {
            return Err(CudaError::Verification { index });
        }
    }
    Ok(ProbeReport {
        device: device.expect("nonzero repetitions"),
        repetitions: repetitions.get(),
        records_per_repetition: input.len(),
        verified: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_checks_do_not_allocate_or_need_a_device() {
        assert_eq!(checked_bytes(0), Ok(0));
        assert_eq!(checked_bytes(257), Ok(6168));
        assert_eq!(checked_bytes(usize::MAX), Err(CudaError::Capacity));
        if usize::BITS > 32 {
            assert_eq!(
                checked_bytes(u32::MAX as usize + 1),
                Err(CudaError::Capacity)
            );
        }
    }
}
