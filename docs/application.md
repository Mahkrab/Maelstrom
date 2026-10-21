# Controlling a simulation worker

`maelstrom::application` is a non-visual library boundary for presentation code.
The worker owns construction, stepping and destruction of its solver and any CUDA
context. Existing headless commands and result/replay schemas are unchanged.
The optional [particle viewer](viewer.md) consumes this boundary. There is no image-export command yet.

```rust,no_run
use maelstrom::{
    application::{Action, Controller, RunSpec, RunStatus},
    execution::ExecutionMode,
    neighbours::SearchBackend,
    scene::load_resolved_scene,
};
use std::{path::Path, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = load_resolved_scene(Path::new("scenes/falling-block-04.toml"), None)?;
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
        println!("{:?}", outcome.status); // Inspect failure and accepted state too.
    }
    controller.join()?;
    Ok(())
}
```

Start acknowledges successful construction or a terminal setup outcome; inspect
the status and failure rather than assuming `Ok` means completion. Pause/resume
and single-step are asynchronous. A receipt acknowledges the state at the worker
boundary; in an event loop poll `try_result` instead of calling blocking `wait`.
Pause takes effect between timesteps, and single-step is valid only while paused.
A zero-step run completes immediately. Running executes all fixed timesteps as
fast as resources permit, independently of redraw clocks, without interpolation.

There are eight queued command slots. A full queue returns `QueueFull` without
replacing the selected run; retry after processing acknowledgements. Each run
has one coalescing observation slot holding the latest owned accepted snapshot,
effective settings, cumulative warnings and terminal failure. Presentation can
skip observations but never numerical work. Retain an observation to keep its
snapshot. This slot cannot supply a lossless offline frame sequence.

`start` also resets/restarts: supply the original or changed initial request and
retain the old handle to inspect its outcome. A new generation cancels obsolete
work, while old handles remain isolated. Inputs should be resolved and validated
before replacing a selected run. Worker construction can still fail, for example
when an explicitly selected CUDA device is unavailable. No fallback is performed;
the previous handle remains readable. Ordinary control cannot resume a terminal
or superseded run. Replay reruns initial inputs; it does not resume a snapshot.

`run.cancel()` is latched and bypasses queue capacity. `request_shutdown()` stops
admission, cancels work and wakes an idle/paused worker. Keep the event loop alive
while polling `is_finished()`, then `join()` to confirm teardown. `join()` and the
destructor wait for synchronous solver stages; no shutdown deadline is promised.
A wait timeout is not a terminal result. Failures and final state survive through
retained handles after joining. Unexpected worker panic is reported by join;
process abort or forced termination cannot guarantee retained output.
