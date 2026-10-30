# Selectable PBF workload controls

The application now exposes the validated half-rate policy through
[`--numerical-mode fast`](../../docs/usage.md#numerical-modes), with an explicit
reference fallback. Prefer that selector for application use: it preserves the
scene's other physical inputs. The files and decisions below retain the earlier
experimental comparison and its declared targets, rather than product gates.

These configuration files reproduce the bounded numerical-work comparison. They
use the existing solver and its transactional acceptance, diagnostic and
cancellation paths. No production default or CUDA kernel changes.

**Neither reduced-work policy is qualified for practical fluid use.** The
half-rate control trades speed for excessive compression; the half-iteration
control retains more work and still fails the combined quality screen. Completion
and finite state do not establish acceptable fluid behaviour. The parent report
`docs/project/development/milestones/05/performance-numerical-work.md` owns the
measurements, physical limitations and GO/NO-GO recommendation.

| File | Hz | Jacobi iterations | Correction scale | Relaxation scale |
|---|---:|---:|---:|---:|
| `production-control.toml` | 480 | 8 | 0.0625 | 20 |
| `half-rate.toml` | 240 | 8 | 0.5 | 2 |
| `half-iterations.toml` | 480 | 4 | 0.5 | 2 |

The last two policies also differ from production in correction and
regularisation. Their measured production speedups are numerical-policy
trade-offs. The separate 120/240/480 Hz × 2/3/4/8 matrix holds correction at 0.5
and relaxation at 2 to isolate the scheduling axes.

Select these files with `--config` on a **schema-2 scene**, keeping refinement at
one. Refinement divides timestep by its factor and correction by the factor
squared; selecting another factor would change the experiment. CFL remains an
explicit warning and never silently inserts steps. The old schema-1 bundled
scenes need migration or an explicitly recorded `--set` invocation instead.

These files select the default physics used by the generated benchmark scenes.
`--config` replaces a scene's configuration file. For other physical inputs, keep
their owning configuration and apply the four numerical values with `--set`.

For example, from the EPQ root, these commands generate new scene inputs and
start a new five-second simulation with 8,000 particles. Choose a fresh input
directory. They do not continue a saved simulation.

Use a release build with the `cuda` feature on a supported GPU machine.

```sh
mkdir /tmp/maelstrom-numerical-inputs
python3 tools/performance/frontier_inputs.py /tmp/maelstrom-numerical-inputs
Maelstrom/target/release/maelstrom run \
  /tmp/maelstrom-numerical-inputs/scenes/resolved.toml \
  --config Maelstrom/experiments/numerical-work/half-rate.toml \
  --backend cuda --device 0 --block-size 128 --refine-timestep 1 \
  --steps 1200 --unpaced --measure
```

For 110,592 particles select `scenes/large.toml` from that generated directory.
For either 480 Hz file use 2,400 steps for five simulated seconds. The run command
measures solver calls; it does not reproduce the separately retained repeated
60 Hz observation comparison or the independent trajectory audits.
