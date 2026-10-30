# Built-in scene packages

Every built-in is a self-contained directory with `scene.toml` (geometry and run
limit), `config.toml` (schema-2 fast numerical inputs) and `render.toml` (presentation).
The build embeds all three; installed binaries never read the source checkout.
Use `maelstrom scenes list`, then `maelstrom view builtin:NAME` to start paused or
`maelstrom compile builtin:NAME --output DIR` to start a fresh three-second simulation.

All packages use fit-camera over the finite container and kinetic energy per particle
in joules. Colour intervals are fixed in each render document, so colours remain
comparable between frames rather than being rescaled to each frame's maximum.
The interval is chosen from the scene's particle mass/height scale, not a claim of
physical accuracy. At the default fast rate, `step_stride = 4` gives 60 Hz output.
Reference mode remains available; use stride 8 for 60 Hz at its 480 Hz step rate.

| Selector | Particles | Initial condition and boundaries |
| --- | ---: | --- |
| `falling-block-04` | 64 | Small drop for quick checks |
| `falling-block-06` | 216 | Compact drop |
| `falling-block-08` | 512 | Medium drop |
| `falling-block-10` | 1,000 | Standard drop |
| `falling-block-12` | 1,728 | Larger drop |
| `falling-block-16` | 4,096 | Dense drop |
| `falling-block-20` | 8,000 | Original performance workload size |
| `falling-block-30` | 27,000 | Wider, taller container |
| `falling-block-10-playground` | 1,000 | Compact editable starting point |
| `two-blocks` | 3,456 | Two opposing finite blocks in a wide box |
| `four-blocks` | 4,000 | Blocks converge from four directions and different heights |
| `dam-break` | 5,184 | Tall released slab spreads across a long basin |
| `opposing-jets` | 1,008 | Finite narrow columns with opposing velocity; no continuous inlet |
| `slope-slide` | 1,680 | Inclined plane inside a bounding box |
| `v-channel` | 2,400 | Two inclined half-spaces form a longitudinal channel |
| `shallow-tank` | 3,360 | Shallow body with initial horizontal momentum |
| `large-impact` | 110,592 | Dense block, intended for the CUDA backend |

The planes constrain the intersection of permitted half-spaces. They do not represent
internal solid obstacles or moving walls. Every particle group shares one fluid model
and inherits spacing from `config.toml`; identities are assigned deterministically.
CUDA runs automatically use the retained optimisations, and compilation uses the
parallel encoding pipeline and canonical streaming provenance hashing.

To edit a built-in without changing the installation, run:

```sh
maelstrom scenes create my-channel --from builtin:v-channel
maelstrom view user:my-channel
```

The old flat inputs are byte-preserved under `reference/scenes`, with frozen companion
configurations in `reference/config`. They retain the numerical regression contract
and are not catalogue entries. Historical manifests and source exports keep their
original recorded paths; the regression reader maps those paths to the relocated fixtures.
