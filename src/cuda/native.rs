//! The only Rust unsafe boundary. Transfers complete their host borrows before
//! returning; submissions retain only context-owned device storage. Handles stay
//! on their owning thread, with completion at diagnostics and host observations.

use super::solver::Settings;
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
    fn maelstrom_cuda_storage(
        handle: *mut c_void,
        slot: u32,
        host: *mut c_void,
        bytes: usize,
        action: u32,
        offset: usize,
    ) -> Status;
    fn maelstrom_cuda_solver(
        handle: *mut c_void,
        settings: Settings,
        operation: u32,
        threads: u32,
    ) -> Status;
    fn maelstrom_cuda_checked_solver(
        handle: *mut c_void,
        settings: Settings,
        operation: u32,
        threads: u32,
        diagnostic: *mut u64,
    ) -> Status;
    fn maelstrom_cuda_observe(
        handle: *mut c_void,
        settings: Settings,
        operation: u32,
        threads: u32,
        records: *mut u64,
    ) -> Status;
    fn maelstrom_cuda_prefix(
        handle: *mut c_void,
        settings: Settings,
        threads: u32,
        count_rows: u32,
        diagnostic: *mut u64,
    ) -> Status;
    fn maelstrom_cuda_begin_attempt(handle: *mut c_void) -> Status;
    fn maelstrom_cuda_accept(handle: *mut c_void) -> Status;
    fn maelstrom_cuda_checked_group(
        handle: *mut c_void,
        settings: Settings,
        first_operation: u32,
        threads: u32,
        diagnostic: *mut u64,
    ) -> Status;
}

// Only these private, explicit layouts can cross the solver transfer boundary.
// Every bit pattern is valid and no value contains references or padding.
pub(super) trait Transfer: Copy + Default {}
impl Transfer for [f32; 3] {}
impl Transfer for [f32; 4] {}
impl Transfer for [f32; 8] {}
impl Transfer for f32 {}
impl Transfer for u32 {}
impl Transfer for u64 {}

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
    pub(super) fn accept(&mut self) -> Result<(), CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: This unique thread-local handle has no outstanding work or
        // borrowed host memory. Simulation calls only after final validation.
        // Native code rejects poisoned/closed attempts before exchanging owned
        // vector allocations and their capacities; it makes no driver calls.
        unsafe { maelstrom_cuda_accept(handle.as_ptr()) }.result()
    }

    pub(super) fn begin_attempt(&mut self) -> Result<(), CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: The unique live handle owns the diagnostic allocation. Native
        // code orders any required reset before subsequent kernels and rejects reuse
        // after uncertain driver faults. No host storage is borrowed.
        unsafe { maelstrom_cuda_begin_attempt(handle.as_ptr()) }.result()
    }

    pub(super) fn reserve(&mut self, slot: u32, bytes: usize) -> Result<(), CudaError> {
        if bytes > isize::MAX as usize {
            return Err(CudaError::Capacity);
        }
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: The unique thread-local handle is live. Action zero allocates
        // context-owned storage and never dereferences the null host pointer.
        unsafe { maelstrom_cuda_storage(handle.as_ptr(), slot, std::ptr::null_mut(), bytes, 0, 0) }
            .result()
    }

    pub(super) fn upload<T: Transfer>(&mut self, slot: u32, values: &[T]) -> Result<(), CudaError> {
        let bytes = std::mem::size_of_val(values);
        self.reserve(slot, bytes)?;
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: The private transfer trait permits only C-compatible plain
        // numbers. Action one reads exactly the borrowed slice, synchronously.
        unsafe {
            maelstrom_cuda_storage(
                handle.as_ptr(),
                slot,
                values.as_ptr().cast_mut().cast(),
                bytes,
                1,
                0,
            )
        }
        .result()
    }

    pub(super) fn download<T: Transfer>(
        &mut self,
        slot: u32,
        count: usize,
    ) -> Result<Vec<T>, CudaError> {
        self.download_at(slot, 0, count)
    }

    pub(super) fn download_at<T: Transfer>(
        &mut self,
        slot: u32,
        offset: usize,
        count: usize,
    ) -> Result<Vec<T>, CudaError> {
        let offset = offset
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(CudaError::Capacity)?;
        let bytes = count
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(CudaError::Capacity)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| CudaError::HostAllocation)?;
        values.resize(count, T::default());
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: Action two writes only the allocated, initialised plain-number
        // slice and finishes before returning. Native code checks offset and length
        // against slot capacity before forming the device address.
        unsafe {
            maelstrom_cuda_storage(
                handle.as_ptr(),
                slot,
                values.as_mut_ptr().cast(),
                bytes,
                2,
                offset,
            )
        }
        .result()?;
        Ok(values)
    }

    pub(super) fn launch(
        &mut self,
        settings: Settings,
        operation: u32,
        threads: u32,
    ) -> Result<(), CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        // SAFETY: Settings has the asserted C layout. The bridge checks launch
        // dimensions and all slot sizes. Submission retains only context-owned
        // device allocations; later diagnostics/transfers drain the private stream.
        unsafe { maelstrom_cuda_solver(handle.as_ptr(), settings, operation, threads) }.result()
    }

    pub(super) fn checked_launch(
        &mut self,
        settings: Settings,
        operation: u32,
        threads: u32,
    ) -> Result<u64, CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        let mut diagnostic = u64::MAX;
        // SAFETY: The unique thread-local handle is live and Settings has its
        // asserted C layout. Native code validates storage and dimensions, drains
        // the kernel and completes the eight-byte copy before this borrow ends.
        unsafe {
            maelstrom_cuda_checked_solver(
                handle.as_ptr(),
                settings,
                operation,
                threads,
                &mut diagnostic,
            )
        }
        .result()?;
        Ok(diagnostic)
    }

    pub(super) fn checked_group(
        &mut self,
        settings: Settings,
        first_operation: u32,
        threads: u32,
    ) -> Result<u64, CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        let mut diagnostic = u64::MAX;
        // SAFETY: The unique live handle owns all device scratch. Native code
        // accepts only the fixed Jacobi/projection groups, validates layouts and
        // sizes, drains submitted work, and finishes writing this result before
        // returning. No borrowed host pointer escapes the call.
        unsafe {
            maelstrom_cuda_checked_group(
                handle.as_ptr(),
                settings,
                first_operation,
                threads,
                &mut diagnostic,
            )
        }
        .result()?;
        Ok(diagnostic)
    }

    pub(super) fn observe(
        &mut self,
        settings: Settings,
        operation: u32,
        threads: u32,
    ) -> Result<[u64; 2], CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        let mut records = [u64::MAX; 2];
        // SAFETY: Settings has the checked C layout and records owns two u64s.
        // Native code accepts only acceleration/prediction, validates scratch,
        // completes the stream and copies results before ending this borrow.
        // Summaries remain in context-owned scratch until a synchronous download.
        unsafe {
            maelstrom_cuda_observe(
                handle.as_ptr(),
                settings,
                operation,
                threads,
                records.as_mut_ptr(),
            )
        }
        .result()?;
        Ok(records)
    }

    pub(super) fn prefix(
        &mut self,
        settings: Settings,
        threads: u32,
        count_rows: bool,
    ) -> Result<u64, CudaError> {
        let handle = self.handle.ok_or(CudaError::Poisoned)?;
        let mut record = u64::MAX;
        // SAFETY: The unique handle owns offsets and scratch. Native code checks
        // capacities and launch sizes, orders count/scan kernels on its stream,
        // and completes diagnostic readback before this scalar borrow ends.
        // Without counting, slot 13 supplies count+1 checked row lengths with
        // zero first; malformed values are diagnosed before offsets are used.
        unsafe {
            maelstrom_cuda_prefix(
                handle.as_ptr(),
                settings,
                threads,
                u32::from(count_rows),
                &mut record,
            )
        }
        .result()?;
        Ok(record)
    }

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
