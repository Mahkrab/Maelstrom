# Isolated DFSPH prototype

This executable tests the published DFSPH density/divergence solves using Maelstrom's unchanged CUDA neighbour search. It is separate from Cargo, `Simulation`, the production PBF kernels and the application. Its physical qualification failed; it is not a backend to enable in production.

`prototype.cu` implements the scalar equations in Bender and Koschier (2017), as implemented by SPlisHSPlasH at commit `7586ffa25948c6ce3bc6c238cda30ad8d35f60a0`. `reference.cpp` drives that implementation independently. The attribution/licence is retained in `reference-LICENSE`. The parent [assessment](../../../docs/project/development/milestones/05/dfsph-prototype.md) owns the measurements, criteria and integration decision.

## Build and run

From the Maelstrom directory, build into a new durable experiment directory:

```sh
nvcc --std=c++17 --ftz=false --prec-div=true --prec-sqrt=true -O3 \
  --gpu-architecture=compute_89 --fmad=false \
  experiments/dfsph/prototype.cu -o /absolute/experiment/dfsph-gpu
```

The inputs in the retained evidence are initial particle states, not restart checkpoints. These commands start new simulations. For a two-second cubic tank run at 240 Hz:

```sh
/absolute/experiment/dfsph-gpu /absolute/experiment/inputs/hydro.bin \
  0.004166666666666667 480 1,1,1 0,-9.81,0 0.1 0.1 \
  /absolute/experiment/hydro 0.016666666666666666 0.025 solid
```

Arguments are input file, timestep, steps, physical box upper corner, acceleration, density tolerance in percent, divergence tolerance in percent, output prefix, observation period, collision inset and wall mode. The optional final arguments `wendland 0.15` select the larger-support control; an additional `0.1` selects its under-relaxed Jacobi repair. Otherwise the reference Jacobi weight is 0.5. Warmstart factors remain 0.5 in both cases. These are the bounded controls investigated, not a general parameter interface.

Use `benchmark` instead of the observation period for timing repeats. It still allocates and downloads host state at 60 Hz, retains every observation's fingerprint and saves complete initial, first-step and final states. Qualification runs retain all sampled states. Computation and host-observation times are separate; file writing and fingerprinting are outside those clocks. JSONL traces retain every step's actual timestep, iterations, residuals, row work, kinetic energy, maximum speed, pressure impulse and wall contacts/energy loss.

## Model and boundaries

The bounded model is single-phase, inviscid, with spacing 0.05 m, geometric particle mass 0.125 kg and physical density 1,000 kg/m³. Cubic support is 0.1 m; Wendland C2 support is 0.15 m. Changing support does not change spacing, mass, collision extent or physical box. There is no artificial pressure, viscosity, XSPH or velocity damping.

Solid samples fill cells outside the physical rectangular box to at least one support radius. Each sample has geometric volume `spacing³`. The DFSPH boundary equations are the reference's sampled-volume equations; the single-surface-layer pseudo-volume calibration is not used. Fluid collisions constrain centres to half a spacing inside the box, with zero normal restitution and unchanged tangential velocity. `none` omits solid samples and is used for the unforced shear and compression controls.

The binary input uses the existing 96-byte settings header followed by float3 positions, float3 velocities and recorded planes. The prototype validates the byte count and finite particle state. It consumes count and baseline support from the header; timestep/gravity are explicit arguments, mass is geometric, and box geometry is explicit. Recorded PBF mass, correction parameters and planes do not silently configure DFSPH. This input adapter is for the retained ascending-index fixtures, not the public scene-loading interface.

Neighbour lists use the retained cell sort, device prefix and ordered rows. Storage grows from measured counts; it has no 512-neighbour-per-row assumption. Boundaries participate in discovery, while pressure unknowns belong only to fluid particles. The density/factor pass is fused; each pressure iteration computes acceleration and its density response from immutable launch snapshots. Mean positive residuals and deficiency handling follow the scalar reference. They are not substitutes for independent physical tests.

## Reference and validation

Build SPlisHSPlasH privately at the pinned commit with `SPH_LIBS_ONLY=ON`, single precision, CompactNSearch and AVX disabled. The retained build metadata names every dependency commit and private Eigen include path. It also records the two required include-only portability repairs. Link `reference.cpp` against that build; no global installation is required.

The reference command takes the legacy input, timestep, steps, three acceleration components, box, wall mode, prefix, density/divergence percentages and collision inset; optional `wendland` selects 0.15 m support. The harness explicitly sets fluid volume and mass after the reference's radius-derived defaults, so support refinement cannot change material mass. The repaired reference library changes only the two Jacobi update weights from 0.5 to 0.1; the exact patch and build commands are retained.

Parent tooling provides campaign collection, independent audits, paired benchmarks, workload counters and rejection tests. The independent readers use binary64 current-position quadrature and recorded boundary volumes. The PBF workload counter is a separate endpoint-matched replay; its clocks are not production throughput evidence.

The prototype has no production cancellation/rollback or checkpoint API. Failure exits preserve diagnostic files rather than offering continued execution. All production interfaces and numerical behaviour remain unchanged. The tested physical and workload failures must be resolved before integration work is justified.
