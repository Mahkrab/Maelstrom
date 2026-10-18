//! The only Rust unsafe boundary. Native calls borrow slices synchronously and
//! opaque handles are neither shared nor moved across threads.

use super::{CudaError, CudaStage, DeviceInfo, Probe};
use std::{
    ffi::{c_char, c_void},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

#[repr(C)]
struct Status {
    code: i32,
    stage: i32,
}

impl Status {
    fn result(self) -> Result<(), CudaError> {
        if self.code == 0 {
            return Ok(());
        }
        if self.stage == 14 {
            return Err(CudaError::Poisoned);
        }
        let stage = match self.stage {
            1 => CudaStage::DriverLoad,
            2 => CudaStage::Initialise,
            3 => CudaStage::DeviceSelection,
            4 => CudaStage::Context,
            5 => CudaStage::Module,
            6 => CudaStage::Stream,
            7 => CudaStage::Allocation,
            8 => CudaStage::Upload,
            9 => CudaStage::Launch,
            10 => CudaStage::Completion,
            11 => CudaStage::Download,
            12 => CudaStage::Release,
            13 => CudaStage::ContextBinding,
            _ => CudaStage::Close,
        };
        Err(CudaError::Driver {
            stage,
            code: self.code,
            invalidated: !(self.stage <= 6 || (self.stage == 7 && self.code == 2)),
        })
    }
}

#[repr(C)]
struct NativeInfo {
    name: [c_char; 128],
    memory_bytes: u64,
    major: i32,
    minor: i32,
    driver_version: i32,
}

unsafe extern "C" {
    fn maelstrom_cuda_open(
        ordinal: i32,
        ptx: *const u8,
        target: i32,
        info: *mut NativeInfo,
        out: *mut *mut c_void,
    ) -> Status;
    fn maelstrom_cuda_roundtrip(
        handle: *mut c_void,
        input: *const Probe,
        output: *mut Probe,
        count: usize,
    ) -> Status;
    fn maelstrom_cuda_close(handle: *mut c_void) -> Status;
}

pub(super) struct Context {
    handle: Option<NonNull<c_void>>,
    pub info: DeviceInfo,
    _thread: PhantomData<Rc<()>>,
}

const _: () = {
    assert!(size_of::<Status>() == 8 && align_of::<Status>() == 4);
    assert!(size_of::<NativeInfo>() == 152 && align_of::<NativeInfo>() == 8);
    assert!(std::mem::offset_of!(NativeInfo, memory_bytes) == 128);
    assert!(std::mem::offset_of!(NativeInfo, driver_version) == 144);
};

impl Context {
    pub fn new(ordinal: u32) -> Result<Self, CudaError> {
        let mut info = NativeInfo {
            name: [0; 128],
            memory_bytes: 0,
            major: 0,
            minor: 0,
            driver_version: 0,
        };
        let mut handle = std::ptr::null_mut();
        const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/roundtrip.ptx"));
        let arch: i32 = env!("MAELSTROM_CUDA_ARCH")
            .parse()
            .expect("build-validated architecture");
        // SAFETY: PTX is NUL-terminated by build.rs. Both writable output objects
        // live for the call. Native construction publishes a handle only on success
        // and destroys partially constructed resources on all failure paths.
        let status = unsafe {
            maelstrom_cuda_open(ordinal as i32, PTX.as_ptr(), arch, &mut info, &mut handle)
        };
        if status.stage == 3 && status.code == 101 {
            return Err(CudaError::InvalidDevice { ordinal });
        }
        status.result()?;
        let handle = NonNull::new(handle).ok_or(CudaError::Driver {
            stage: CudaStage::Context,
            code: -5,
            invalidated: true,
        })?;
        let name: Vec<u8> = info
            .name
            .iter()
            .take_while(|&&byte| byte != 0)
            .map(|&byte| byte as u8)
            .collect();
        Ok(Self {
            handle: Some(handle),
            info: DeviceInfo {
                ordinal,
                name: String::from_utf8_lossy(&name).into_owned(),
                memory_bytes: info.memory_bytes,
                compute_major: info.major,
                compute_minor: info.minor,
                driver_version: info.driver_version,
                target: format!("compute_{arch}"),
            },
            _thread: PhantomData,
        })
    }

    pub fn roundtrip(&mut self, input: &[Probe]) -> Result<Vec<Probe>, CudaError> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(input.len())
            .map_err(|_| CudaError::HostAllocation)?;
        output.resize(input.len(), Probe::default());
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: Both slices have the same validated length and explicit C layout.
        // Native transfers are synchronous; kernel completion precedes download.
        // The unique handle is live, thread-local and exclusively borrowed here.
        unsafe {
            maelstrom_cuda_roundtrip(
                handle.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
            )
        }
        .result()?;
        Ok(output)
    }

    pub fn close(mut self) -> Result<(), CudaError> {
        if let Some(handle) = self.handle.take() {
            // SAFETY: Taking the unique live handle prevents subsequent calls and
            // Drop from releasing it twice. Native close consumes it even on error.
            unsafe { maelstrom_cuda_close(handle.as_ptr()) }.result()?;
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // SAFETY: This is the only remaining owner; no operation is pending
            // on borrowed host memory and the handle is consumed exactly once.
            let _ = unsafe { maelstrom_cuda_close(handle.as_ptr()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_status_preserves_submission_completion_and_retry_policy() {
        for (stage, expected) in [
            (7, CudaStage::Allocation),
            (8, CudaStage::Upload),
            (9, CudaStage::Launch),
            (10, CudaStage::Completion),
            (11, CudaStage::Download),
            (12, CudaStage::Release),
            (13, CudaStage::ContextBinding),
            (15, CudaStage::Close),
        ] {
            assert_eq!(
                Status { code: 999, stage }.result(),
                Err(CudaError::Driver {
                    stage: expected,
                    code: 999,
                    invalidated: true
                })
            );
        }
        assert_eq!(
            Status { code: 2, stage: 7 }.result(),
            Err(CudaError::Driver {
                stage: CudaStage::Allocation,
                code: 2,
                invalidated: false
            })
        );
        assert_eq!(
            Status {
                code: -3,
                stage: 14
            }
            .result(),
            Err(CudaError::Poisoned)
        );
        assert_eq!(Status { code: 0, stage: 0 }.result(), Ok(()));
    }
}
