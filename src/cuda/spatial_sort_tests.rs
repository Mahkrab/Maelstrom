//! Raw cell keys exercise the complete signed ordering independently of floating
//! point cell construction. Padding duplicates must survive every merge boundary.

use super::*;

#[test]
#[ignore = "requires an explicitly selected CUDA hardware test run"]
fn hardware_spatial_sort_preserves_full_keys_padding_and_reused_storage() {
    let config = crate::config::resolve_effective_configuration(None, None, None).unwrap();
    for threads in [32, 64, 128, 256] {
        let mut gpu = CudaSolver::new(0, threads).unwrap();
        for count in [0, 1, 2, 127, 128, 129, 257, 1025, 4097, 129] {
            let particles = vec![
                Particle {
                    identity: 0,
                    position_m: Vec3::ZERO,
                    velocity_m_s: Vec3::ZERO,
                };
                count
            ];
            let s = settings(&config, count, 0).unwrap();
            gpu.prepare(s, &particles, &[]).unwrap();
            let coordinates = [i64::MIN, -(1_i64 << 52), -1, 0, 1, 1_i64 << 52, i64::MAX];
            let mut keys: Vec<_> = (0..count)
                .map(|i| {
                    (
                        coordinates[(i * 13) % coordinates.len()],
                        coordinates[(i / 3) % coordinates.len()],
                        coordinates[(i / 11) % coordinates.len()],
                        i as u32,
                    )
                })
                .collect();
            keys.reverse();
            let mut packed: Vec<u64> = keys
                .iter()
                .flat_map(|&(x, y, z, index)| [x as u64, y as u64, z as u64, u64::from(index)])
                .collect();
            let sentinel = [
                i64::MAX as u64,
                i64::MAX as u64,
                i64::MAX as u64,
                u32::MAX as u64,
            ];
            for _ in count..s.padded as usize {
                packed.extend(sentinel);
            }
            gpu.context.inner.upload(10, &packed).unwrap();
            gpu.context.inner.launch(s, 13, threads).unwrap();
            let sorted: Vec<u64> = gpu.context.inner.download(10, packed.len()).unwrap();
            keys.sort_unstable();
            let actual: Vec<_> = sorted
                .chunks_exact(4)
                .take(count)
                .map(|c| (c[0] as i64, c[1] as i64, c[2] as i64, c[3] as u32))
                .collect();
            assert_eq!(actual, keys, "count {count}, block {threads}");
            for padding in sorted.chunks_exact(4).skip(count) {
                assert_eq!(padding, sentinel);
            }
        }
    }
}
