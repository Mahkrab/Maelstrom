//! Missing columns and widely separated z-cells expose range scans which cross
//! a cell boundary incorrectly. CPU all-pairs discovery supplies the row oracle.

use super::{CudaSolver, tests::discovered};
use crate::{math::Vec3, neighbours::brute_force};

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_discovery_keeps_discontinuous_columns_and_identity_order() {
    let mut positions = Vec::new();
    for x in -2_i32..=2 {
        for y in -2_i32..=2 {
            for z in -5_i32..=5 {
                if (x + y + z).rem_euclid(4) == 0 {
                    continue;
                }
                for fraction in [0.01, 0.0625, 0.115] {
                    positions.push(Vec3::new(
                        x as f32 * 0.125 + fraction,
                        y as f32 * 0.125 + fraction,
                        z as f32 * 0.125 + fraction,
                    ));
                }
            }
        }
    }
    // Spatial order must not accidentally serve as identity order. Include
    // duplicate points and signed zero without changing strict support rules.
    positions.reverse();
    positions.extend([Vec3::ZERO, Vec3::new(-0.0, 0.0, 0.0)]);
    assert!(positions.len() > 512 && !positions.len().is_multiple_of(32));
    let expected = brute_force(&positions, 0.1).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        let rows = discovered(&mut gpu, &positions, 0.1).unwrap();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row,
                expected.for_particle(i).unwrap(),
                "block {threads}, row {i}"
            );
        }
        // Reusing storage with missing neighbour columns must clear old rows.
        let sparse = [
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 100.0),
            Vec3::new(0.0, 100.0, 0.0),
        ];
        assert_eq!(
            discovered(&mut gpu, &sparse, 0.1).unwrap(),
            vec![Vec::<usize>::new(); 3]
        );
        assert_eq!(
            discovered(&mut gpu, &[], 0.1).unwrap(),
            Vec::<Vec<usize>>::new()
        );
    }
}
