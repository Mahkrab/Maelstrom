//! Complete public Simulation timestep timing, separate from native attribution.
//! Each process starts a new simulation; accepted warm-up steps are untimed.
use maelstrom::{
    config::{parse_configuration_overlay, resolve_effective_configuration},
    execution::ExecutionMode,
    math::Vec3,
    neighbours::SearchBackend,
    scene::Particle,
    simulation::Simulation,
};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: compact_grid_step_bench PARTICLES normal|dense|sparse STEPS [WARMUP_STEPS]"
                .into(),
        );
    }
    let count = args[1].parse::<usize>()?;
    if count == 0 || count > 1 << 26 {
        return Err("particle count outside benchmark bound".into());
    }
    let spacing = match args[2].as_str() {
        "normal" => 0.05,
        "dense" => 0.035,
        "sparse" => 0.4,
        _ => return Err("unknown distribution".into()),
    };
    let steps = args[3].parse::<usize>()?;
    let overlay = parse_configuration_overlay("[numerical]\nrefine_timestep=4")?;
    let config = resolve_effective_configuration(None, None, Some(&overlay))?;
    let side = (count as f64).cbrt().ceil() as usize;
    let particles: Vec<_> = (0..count)
        .map(|i| Particle {
            identity: i as u64,
            position_m: Vec3::new(
                (i % side) as f32 * spacing,
                (i / side % side) as f32 * spacing + 1.0,
                (i / (side * side)) as f32 * spacing,
            ),
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    let mut simulation = Simulation::new_with_execution(
        &config,
        particles,
        vec![],
        SearchBackend::UniformGrid,
        ExecutionMode::Cuda {
            device: 0,
            threads: 128,
        },
    )?;
    let warmup = args
        .get(4)
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if warmup == 0 {
        return Err("warm-up must include an accepted step".into());
    }
    for _ in 0..warmup {
        let _ = simulation.step()?;
    }
    let profile = std::env::var_os("MAELSTROM_GRID_BENCH_PROFILE").is_some();
    let mut outcomes = Vec::new();
    let mut times = Vec::new();
    let mut stage_times = Vec::new();
    for _ in 0..steps {
        let (outcome, measurement) =
            simulation.step_measured_with_cancellation(profile, || false)?;
        outcomes.push(outcome);
        times.push(measurement.total_s);
        stage_times.push(measurement.stages);
    }
    // Final state hashing is outside solver clocks. Compare matched policies
    // including every position and velocity, independent of timing variability.
    let mut hash = Sha256::new();
    for particle in simulation.snapshot().particles() {
        hash.update(particle.identity.to_le_bytes());
        for value in [
            particle.position_m.x,
            particle.position_m.y,
            particle.position_m.z,
            particle.velocity_m_s.x,
            particle.velocity_m_s.y,
            particle.velocity_m_s.z,
        ] {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    let digest: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let mut diagnostics = Sha256::new();
    for outcome in outcomes {
        diagnostics.update(outcome.completed_steps.to_le_bytes());
        diagnostics.update(outcome.cfl.maximum_speed.to_bits().to_le_bytes());
        match outcome.cfl.limit {
            maelstrom::diagnostics::CflLimit::Unbounded => diagnostics.update([0]),
            maelstrom::diagnostics::CflLimit::Finite(value) => {
                diagnostics.update([1]);
                diagnostics.update(value.to_bits().to_le_bytes());
            }
        }
        diagnostics.update([u8::from(outcome.cfl.warning)]);
    }
    let diagnostics_digest: String = diagnostics
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "policy": std::env::var("MAELSTROM_CUDA_SPATIAL_INDEX").unwrap_or_else(|_| "grid".into()),
            "particles": count, "distribution": args[2], "warmup_steps": warmup, "steps_s": times, "stage_times": stage_times,
            "accepted_state_sha256": digest, "diagnostics_sha256": diagnostics_digest, "solver_iterations": config.numerical.solver_iterations,
            "timestep_s": config.numerical.fixed_timestep_s, "support_radius_m": config.derived.support_radius_m,
            "source_sha256": env!("MAELSTROM_EXPORT_SOURCE"),
        })
    );
    Ok(())
}
