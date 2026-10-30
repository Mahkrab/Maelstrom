# Developer application API

The CLI is described in [usage](usage.md). This page is for callers embedding the
non-visual `maelstrom::application` controller. Its worker owns solver construction,
stepping and teardown, including any thread-bound CUDA context. Presentation receives
owned accepted observations; it cannot mutate the solver or decide step acceptance.

## Start, observe and control

```rust,no_run
use maelstrom::{
    application::{Action, Controller, RunSpec, RunStatus},
    execution::ExecutionMode,
    neighbours::SearchBackend,
    scene::load_resolved_scene,
};
use std::{path::Path, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = load_resolved_scene(Path::new("scenes/falling-block-04/scene.toml"), None)?;
    let mut controller = Controller::new()?;
    let (run, started) = controller.start(RunSpec {
        scene,
        backend: SearchBackend::UniformGrid,
        execution: ExecutionMode::Serial,
        steps: 120,
        initially_paused: true,
    })?;
    assert_eq!(started.wait()?, RunStatus::Paused);
    controller.control(&run, Action::SingleStep)?.wait()?;
    assert_eq!(run.observe().snapshot.unwrap().completed_steps(), 1);
    controller.control(&run, Action::Resume)?.wait()?;
    if let Some(outcome) = run.wait_for_terminal(Duration::from_secs(30)) {
        println!("{:?}", outcome.status);
    }
    controller.join()?;
    Ok(())
}
```

The start receipt acknowledges construction or terminal setup failure, not successful
completion. Inspect status and failure. Control receipts describe a worker boundary;
event loops should poll `try_result` rather than block in `wait`.
Single-step requires pause. Pause takes effect between steps; a zero-step request ends
at zero. The worker advances fixed steps as fast as resources permit, independently of
redraws. Headless CLI pacing is a separate caller concern, not a new solver policy.

## Backpressure and generation ownership

Eight queued command slots bound admission. `QueueFull` leaves the selected request
intact; process receipts before retrying. Each run has one coalescing observation slot
with its latest accepted snapshot, effective resources, warnings and terminal outcome.
Keep an observation to retain its snapshot. Dropping intermediate observations does
not drop numerical work.

Starting another request creates a generation, cancels obsolete work and preserves
old handles as independent readable outcomes. Resolve and validate input before
replacement; resource construction may still fail. Explicit CUDA failure has no
worker-level CPU fallback. A terminal or superseded handle cannot be resumed.
Numerical rerun reconstructs initial input, not a mid-simulation checkpoint.

Compilation cannot use this coalesced slot as a lossless history. The exporter drives
accepted steps synchronously and waits for each scheduled image publication. Both
paths retain the same mathematical and atomic-acceptance boundaries.

## Cancellation and teardown

`run.cancel()` is latched and bypasses queued-command capacity.
`request_shutdown()` stops admission, cancels work and wakes paused/idle workers.
Keep processing events while polling `is_finished()`, then `join()` to confirm teardown.
Joining and destruction can wait for synchronous solver/driver stages; there is no
universal shutdown deadline. A wait timeout is not a terminal simulation result.

Failures and accepted state remain available through retained handles after joining.
Unexpected worker panic is reported by join. Process abort or forced termination
cannot guarantee output. Never add unsafe cross-thread CUDA sharing to bypass the
worker's ownership boundary.
