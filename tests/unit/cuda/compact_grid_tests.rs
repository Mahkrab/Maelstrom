//! The sorted GPU path and independent all-pairs oracle check every ordered row.
use super::*;

#[test]
fn grid_admission_bounds_dimensions_storage_and_sparse_fallback() {
    let mut s = Settings {
        count: 1000,
        padded: 1024,
        radius: 1.0,
        width: 1.0,
        ..Default::default()
    };
    configure_grid(
        &mut s,
        Vec3::new(-2.0, -3.0, -4.0),
        Vec3::new(2.0, 3.0, 4.0),
    );
    assert_eq!(s.grid_min, [-2, -3, -4]);
    assert_eq!(s.grid_dims, [5, 7, 9]);
    assert_eq!(s.grid_cells, 315);
    assert!(storage_sizes(s).unwrap()[10] < 1024 * 64);
    for (lower, upper) in [
        (Vec3::ZERO, Vec3::new(1e20, 0.0, 0.0)),
        (Vec3::ZERO, Vec3::new(1000.0, 1000.0, 1000.0)),
        (Vec3::new(f32::NAN, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
    ] {
        s.grid_cells = 0;
        s.tile_base = 0;
        configure_grid(&mut s, lower, upper);
        assert_eq!(s.grid_cells, 0);
    }
    s.count = 1;
    configure_grid(
        &mut s,
        Vec3::new(-1e15, 0.0, 0.0),
        Vec3::new(-1e15, 0.0, 0.0),
    );
    assert_eq!(s.grid_cells, 1);
    s.grid_cells = 0;
    s.width = 0.5;
    configure_grid(&mut s, Vec3::ZERO, Vec3::ZERO);
    assert_eq!(s.grid_cells, 0, "adjacent cells require width >= radius");
}

fn rows(gpu: &mut CudaSolver, points: &[Vec3], tiled: bool) -> (Settings, Vec<Vec<usize>>) {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let mut s = settings(&config, points.len(), 0).unwrap();
    s.radius = 1.0;
    s.neighbour_layout = u32::from(tiled);
    let particles: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| Particle {
            identity: i as u64,
            position_m: p,
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    gpu.prepare(s, &particles, &[]).unwrap();
    gpu.context
        .inner
        .upload(2, &points.iter().copied().map(array).collect::<Vec<_>>())
        .unwrap();
    gpu.discover(&mut s, points).unwrap();
    let offsets = gpu
        .context
        .inner
        .download::<u64>(13, points.len() + 1)
        .unwrap();
    assert_eq!(offsets[0], 0);
    let indices = gpu
        .context
        .inner
        .download::<u32>(11, s.entries as usize)
        .unwrap();
    let tiles = if tiled && !points.is_empty() {
        gpu.context
            .inner
            .download_at::<u64>(10, tile_base(s), points.len().div_ceil(32) + 1)
            .unwrap()
    } else {
        vec![]
    };
    let result = (0..points.len())
        .map(|i| {
            assert!(offsets[i] <= offsets[i + 1]);
            let row: Vec<_> = (0..(offsets[i + 1] - offsets[i]) as usize)
                .map(|k| {
                    let at = if tiled {
                        tiles[i / 32] as usize + i % 32 + k * 32
                    } else {
                        offsets[i] as usize + k
                    };
                    let j = indices[at] as usize;
                    assert!(j < points.len() && j != i);
                    j
                })
                .collect();
            assert!(
                row.windows(2).all(|pair| pair[0] < pair[1]),
                "duplicates or unsorted row {i}"
            );
            row
        })
        .collect();
    if s.grid_cells != 0 {
        let m = s.grid_cells as usize;
        let counts = gpu.context.inner.download::<u32>(10, m).unwrap();
        let grid_offsets = gpu.context.inner.download_at::<u32>(10, m, m + 1).unwrap();
        assert_eq!(grid_offsets[0], 0);
        assert_eq!(grid_offsets[m] as usize, points.len());
        for i in 0..m {
            assert_eq!(grid_offsets[i + 1] - grid_offsets[i], counts[i]);
        }
        let mut permutation = gpu
            .context
            .inner
            .download_at::<u32>(10, 3 * m + 1 + points.len(), points.len())
            .unwrap();
        permutation.sort_unstable();
        assert_eq!(permutation, (0..s.count).collect::<Vec<_>>());
    }
    (s, result)
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_compact_grid_matches_sorted_and_bruteforce_complete_rows() {
    let below = f32::from_bits(1.0_f32.to_bits() - 1);
    let above = f32::from_bits(1.0_f32.to_bits() + 1);
    let mut cases = vec![
        vec![],
        vec![Vec3::ZERO],
        vec![Vec3::ZERO, Vec3::new(below, 0.0, 0.0)],
        vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)],
        vec![Vec3::ZERO, Vec3::new(above, 0.0, 0.0)],
        vec![Vec3::ZERO; 257],
    ];
    cases.push(
        (-9..=9)
            .flat_map(|x| (-4..=4).map(move |y| Vec3::new(x as f32 * 0.5, y as f32 * 0.5, -1.0)))
            .collect(),
    );
    let mut seed = 0x4d595df4d0f33173_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / 16777216.0 - 0.5) * 8.0
    };
    cases.push(
        (0..513)
            .map(|_| Vec3::new(random(), random(), random()))
            .collect(),
    );
    cases.push(vec![Vec3::new(-1e15, 0.0, 0.0), Vec3::new(-1e15, 0.0, 0.0)]);
    cases.push(vec![
        Vec3::ZERO,
        Vec3::new(1e15, 0.0, 0.0),
        Vec3::new(0.0, -1e15, 0.0),
    ]);
    // >256 scan blocks exercises the chunked upper scan and sparse empty cells.
    cases.push(
        (0..33_001)
            .map(|i| {
                Vec3::new(
                    (i % 41) as f32 * 1.3,
                    (i / 41 % 41) as f32 * 1.3,
                    (i / 1681) as f32 * 1.3,
                )
            })
            .collect(),
    );
    for threads in [32, 64, 128, 256] {
        let mut grid = CudaSolver::new(0, threads).unwrap();
        grid.force_sorted = false;
        let mut sorted = CudaSolver::new(0, threads).unwrap();
        sorted.force_sorted = true;
        for (case, points) in cases.iter().enumerate() {
            let oracle = if points.len() < 2000 {
                crate::neighbours::brute_force(points, 1.0).unwrap()
            } else {
                crate::neighbours::uniform_grid(points, 1.0).unwrap()
            };
            for tiled in [false, true] {
                let (s, actual) = rows(&mut grid, points, tiled);
                assert_eq!(
                    actual,
                    rows(&mut sorted, points, tiled).1,
                    "sorted case {case}, threads {threads}"
                );
                for (i, row) in actual.iter().enumerate() {
                    assert_eq!(
                        row,
                        oracle.for_particle(i).unwrap(),
                        "oracle case {case}, row {i}"
                    );
                }
                for _ in 0..2 {
                    assert_eq!(
                        actual,
                        rows(&mut grid, points, tiled).1,
                        "repeat case {case}"
                    );
                }
                if case == 9 {
                    assert_eq!(s.grid_cells, 0, "pathological sparse fallback");
                } else if !points.is_empty() {
                    assert_ne!(s.grid_cells, 0, "normal case {case} should use grid");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_compact_grid_and_sorted_steps_are_bitwise_identical() {
    let mut config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    config.numerical.xsph_viscosity_enabled = true;
    config.numerical.vorticity_confinement_enabled = true;
    let mut grid = CudaSolver::new(0, 128).unwrap();
    grid.force_sorted = false;
    let mut sorted = CudaSolver::new(0, 128).unwrap();
    sorted.force_sorted = true;
    let mut particles: Vec<_> = (0..513)
        .map(|i| Particle {
            identity: i,
            position_m: Vec3::new(
                (i % 9) as f32 * 0.025,
                (i / 9 % 9) as f32 * 0.025 + 0.1,
                (i / 81) as f32 * 0.025,
            ),
            velocity_m_s: Vec3::new(i as f32 * 0.00001, 0.0, 0.0),
        })
        .collect();
    let planes = [Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0, 1e-6).unwrap()];
    for _ in 0..12 {
        let a = grid
            .step(
                &config,
                &particles,
                &planes,
                &mut || false,
                &mut StageClock::<false>::new(),
            )
            .unwrap();
        let b = sorted
            .step(
                &config,
                &particles,
                &planes,
                &mut || false,
                &mut StageClock::<false>::new(),
            )
            .unwrap();
        let bits = |p: &[Particle]| {
            p.iter()
                .map(|p| {
                    (
                        p.identity,
                        array(p.position_m).map(f32::to_bits),
                        array(p.velocity_m_s).map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a.particles), bits(&b.particles));
        assert_eq!(a.cfl, b.cfl);
        assert_eq!(a.planes_checked, b.planes_checked);
        grid.accept().unwrap();
        sorted.accept().unwrap();
        particles = a.particles;
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_grid_rejects_bad_bounds_and_coordinates_before_neighbour_access() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let particles = vec![
        Particle {
            identity: 0,
            position_m: Vec3::ZERO,
            velocity_m_s: Vec3::ZERO
        };
        3
    ];
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let mut s = settings(&config, 3, 0).unwrap();
    s.radius = 1.0;
    s.width = 1.0;
    configure_grid(&mut s, Vec3::ZERO, Vec3::ZERO);
    gpu.prepare(s, &particles, &[]).unwrap();
    gpu.context
        .inner
        .reserve(10, storage_sizes(s).unwrap()[10])
        .unwrap();
    for x in [2.0, f32::NAN, f32::MAX, 0.0] {
        gpu.context.inner.begin_attempt().unwrap();
        gpu.context
            .inner
            .upload(2, &[[x, 0.0, 0.0], [0.0; 3], [0.0; 3]])
            .unwrap();
        gpu.context.inner.launch(s, 26, 128).unwrap();
        let record = gpu.context.inner.prefix(s, 128, true).unwrap();
        assert_eq!(record, if x == 0.0 { u64::MAX } else { 52 });
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run; scans the admission ceiling"]
fn hardware_grid_scan_at_cell_limit_keeps_exact_offsets_and_permutation() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let count = 128_usize.pow(3);
    let mut points: Vec<_> = (0..count)
        .map(|i| {
            Vec3::new(
                (i % 128) as f32 * 2.0,
                (i / 128 % 128) as f32 * 2.0,
                (i / 16384) as f32 * 2.0,
            )
        })
        .collect();
    points[count - 1] = Vec3::new(255.0, 255.0, 255.0);
    let particles: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, &position_m)| Particle {
            identity: i as u64,
            position_m,
            velocity_m_s: Vec3::ZERO,
        })
        .collect();
    let mut s = settings(&config, count, 0).unwrap();
    s.width = 1.0;
    s.radius = 1.0;
    configure_grid(&mut s, Vec3::ZERO, Vec3::new(255.0, 255.0, 255.0));
    assert_eq!(u64::from(s.grid_cells), MAX_GRID_CELLS);
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    gpu.prepare(s, &particles, &[]).unwrap();
    gpu.context
        .inner
        .reserve(10, storage_sizes(s).unwrap()[10])
        .unwrap();
    gpu.context
        .inner
        .upload(2, &points.iter().copied().map(array).collect::<Vec<_>>())
        .unwrap();
    gpu.context.inner.launch(s, 26, 128).unwrap();
    assert_eq!(
        gpu.context.inner.checked_launch(s, 0, 128).unwrap(),
        u64::MAX
    );
    let m = s.grid_cells as usize;
    let packed = gpu.context.inner.download::<u32>(10, 2 * m + 1).unwrap();
    let (counts, offsets) = packed.split_at(m);
    assert_eq!(offsets[0], 0);
    assert_eq!(offsets[m] as usize, count);
    for cell in 0..m {
        assert_eq!(offsets[cell + 1] - offsets[cell], counts[cell]);
    }
    let indices = gpu
        .context
        .inner
        .download_at::<u32>(10, 3 * m + 1 + count, count)
        .unwrap();
    let mut seen = vec![false; count];
    for (cell, range) in offsets.windows(2).enumerate() {
        for &j in &indices[range[0] as usize..range[1] as usize] {
            let j = j as usize;
            assert!(j < count && !seen[j]);
            seen[j] = true;
            let p = points[j];
            assert_eq!(
                (p.x as usize * 256 + p.y as usize) * 256 + p.z as usize,
                cell
            );
        }
    }
    assert!(seen.into_iter().all(|x| x));
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_grid_fill_errors_keep_lowest_identity_across_blocks() {
    let points: Vec<_> = (0..1025)
        .map(|i| Vec3::new(i as f32 * 0.5, 0.0, 0.0))
        .collect();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        gpu.force_sorted = false;
        for _ in 0..8 {
            let (s, _) = rows(&mut gpu, &points, false);
            let offsets = vec![s.entries + 1; points.len() + 1];
            gpu.context.inner.upload(13, &offsets).unwrap();
            assert_eq!(
                gpu.context.inner.checked_launch(s, 12, threads).unwrap(),
                52
            );
            gpu.context.inner.begin_attempt().unwrap();
            // The next rebuild must clear rejected rows and diagnostics.
        }
    }
}

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_grid_reservation_amortises_small_bounds_growth_within_cap() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    let mut gpu = CudaSolver::new(0, 128).unwrap();
    let mut s = settings(&config, 1024, 0).unwrap();
    s.width = 1.0;
    s.radius = 1.0;
    let mut changes = 0;
    let mut previous = 0;
    for cells in 1000..1500 {
        s.grid_cells = 0;
        s.tile_base = 0;
        configure_grid(&mut s, Vec3::ZERO, Vec3::new(cells as f32 - 1.0, 0.0, 0.0));
        assert_eq!(s.grid_cells, cells);
        gpu.reserve_index(s).unwrap();
        assert!(gpu.index_capacity >= storage_sizes(s).unwrap()[10]);
        assert!(gpu.index_capacity <= max_index_bytes(s).unwrap());
        if gpu.index_capacity != previous {
            changes += 1;
            previous = gpu.index_capacity;
        }
    }
    assert_eq!(
        changes, 2,
        "500 small changes should need only two reservations"
    );
}
