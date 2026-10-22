# Scenes, geometry and configuration

Maelstrom accepts strict TOML inputs with `schema_version = 1`. Unknown fields and unsupported
versions fail rather than acquiring an implicit meaning. Numerical inputs and derived calculations
use binary32; coordinates are right-handed with positive Y upward, in metres, seconds and kilograms.
Names and descriptions must contain non-whitespace text and no control characters.

## Your scene library

Start from an editable copy rather than changing installed examples:

```sh
maelstrom scenes create water --from builtin:falling-block-04
maelstrom paths
maelstrom inspect water --backend serial-grid
maelstrom validate water
```

Each package is `<scenes-root>/water/scene.toml` plus `config.toml`. The scene document
holds particle/lattice geometry and ordered static planes. Its configuration reference
is scene-relative, so moving the package does not depend on the shell's working
directory. Creation copies effective primary numerical values, not inspection-only
derived diagnostics. There is no mesh or arbitrary external geometry format.

Without overrides, scenes live in `$XDG_DATA_HOME/maelstrom/scenes`, defaulting to
`~/.local/share/maelstrom/scenes`. Set absolute `MAELSTROM_SCENES_DIR` to relocate
the library, or pass `--scenes-dir PATH` to override that variable for one command.
Named lookup never searches random working-directory folders. An explicit
`./scene.toml` remains independent of the library. Duplicate user/bundled names
require `user:name` or `builtin:name`; discovery is ordered and does not create files.

Compiled runs are a different library: `$XDG_DATA_HOME/maelstrom/runs`, overridden
by `MAELSTROM_RUNS_DIR` or `--runs-dir`. See [compiled output](export.md).
`maelstrom config show` displays application preferences from
`$XDG_CONFIG_HOME/maelstrom/config.toml`, defaulting to `~/.config/maelstrom/config.toml`.
The preferences schema permits only `schema_version = 1`, `backend`, `threads`,
`device` and `block_size`. Explicit execution selection takes precedence; these
preferences never override scene physics. For example:

```toml
schema_version = 1
backend = "threaded-grid"
threads = 4
```

Storage environment overrides must be absolute. Relative explicit CLI paths are
resolved against the invocation directory. Invalid/relative XDG roots use the normal
HOME fallback. Unknown preference fields and schema versions fail rather than changing
the numerical model implicitly.

## Configuration resolution

A configuration document requires `schema_version`, `name` and `description`. Its optional
`[numerical]` table supplies primary values; its optional `[derived]` table asserts calculated values.
An omitted primary field inherits the earlier layer. Resolution applies:

1. Built-in defaults, currently equivalent numerically to `config/defaults-v4.toml`.
2. The selected configuration document.
3. The scene's `configuration.overrides` overlay.
4. Command-line `--set FIELD=VALUE` overrides and then `--refine-timestep N`.

An external scene's configuration path is relative to that scene's directory. `--config FILE`
replaces the selected base document and retains the scene overlay. Bundled scenes and their base
configuration are embedded in the executable, so named scenes also work outside the checkout.
The build discovers the top-level `scenes/*.toml` and `config/*.toml` files in sorted order;
scene selectors come from filenames without `.toml`. Adding or removing a shipped scene
requires a rebuild, not a Rust registry edit. Filenames use ASCII letters, digits, hyphens
and underscores, with stems of at most 120 characters. Each scene retains its own relative
configuration reference. Personal scenes belong in the editable library and need no rebuild.
Environment variables do not select numerical inputs. `inspect --format json` exposes every
resolved value, derived quantity, field provenance and technical comparison key.

Each layer applies primary overrides, then any refinement, derives quantities and checks its own
derived assertions. Refinement divides the effective timestep by N and the complete position
correction scale by N². Only the resulting values enter replay. It is an explicit numerical variant;
it never makes the timestep adaptive. N is an integer from 1 through 16; factors above four have
not been qualified by the retained fluid-stability studies.

For example, a partial configuration can retain all other defaults:

```toml
schema_version = 1
name = "unforced-example"
description = "Default numerical model without external acceleration"

[numerical]
external_acceleration_m_s2 = [0.0, 0.0, 0.0]
```

## Primary numerical fields

All real values below must be finite, including intermediate derived results. Domain acceptance
does not establish that every admitted combination is a qualified fluid workload. Defaults below
describe the unrefined current configuration; the frozen comparison family explicitly refines it.

| Field in `[numerical]` | Default | Meaning and accepted domain |
| --- | --- | --- |
| `external_acceleration_m_s2` | `[0.0, -9.81, 0.0]` | Three finite acceleration components, m/s² |
| `fixed_timestep_s` | `0.008333334` | Fixed seconds per step, greater than zero |
| `refine_timestep` | Absent | Optional layer operation, integer 1–16; not a resolved primary field |
| `correction_scale` | `1.0` | Dimensionless multiplier on the complete Jacobi correction, greater than zero and at most one |
| `cfl_factor` | `0.4` | Positive dimensionless factor in the warning-only speed diagnostic |
| `particle_spacing_m` | `0.05` | Positive spacing and nominal diameter, metres |
| `support_radius_ratio` | `2.0` | Support radius / spacing, greater than zero and at most eight |
| `physical_rest_density_kg_m3` | `1000.0` | Positive physical rest density, kg/m³ |
| `solver_iterations` | `8` | Fixed integer Jacobi work count, 1–64 |
| `relaxation_scale` | `20.0` | Positive dimensionless scale; relaxation is this value / h² |
| `artificial_pressure_enabled` | `true` | Boolean; disabling it derives zero strength |
| `artificial_pressure_strength_scale` | `0.1` | Non-negative dimensionless scale; enabled strength is this value × h² |
| `artificial_pressure_reference_separation_ratio` | `0.2` | Reference separation / h, strictly between zero and one |
| `artificial_pressure_exponent` | `4.0` | Positive exponent |
| `xsph_viscosity_enabled` | `false` | Only `false` is supported |
| `vorticity_confinement_enabled` | `false` | Only `false` is supported |

The following policy fields each accept only the displayed default string. They are recorded
explicitly so an input cannot silently request another numerical method.

| Field | Accepted value |
| --- | --- |
| `scalar_precision` | `"binary32"` |
| `coordinate_system` | `"right_handed_y_up_metres_seconds_kilograms"` |
| `timestep_policy` | `"fixed"` |
| `density_kernel` | `"poly6"` |
| `correction_kernel` | `"spiky"` |
| `solver_update_policy` | `"jacobi_immutable_snapshot"` |
| `support_policy` | `"strict_less_than"` |
| `boundary_model` | `"static_plane_projection"` |
| `neighbour_rebuild_policy` | `"once_after_prediction"` |
| `particle_order_policy` | `"ascending_particle_identity"` |

## Derived assertions

Every field in `[derived]` is optional and must equal the resolver's binary32 bits exactly, including
signed zero. These are consistency checks, never independent inputs. The resolver derives h from
spacing × support ratio, evaluates an ordered strict-support lattice sum for normalised rest
density, and divides physical rest density by that sum for mass. At the default ratio the sum has
27 offsets. `inspect` also reports `lattice_offset_count`; that output is not an accepted assertion.

| Assertion field | Unit or relationship |
| --- | --- |
| `particle_diameter_m` | Spacing, m |
| `support_radius_m` | h, m |
| `normalised_rest_density_per_m3` | Lattice kernel sum, m⁻³ |
| `particle_mass_kg` | Physical rest density / normalised rest density, kg |
| `relaxation_per_m2` | Relaxation scale / h², m⁻² |
| `artificial_pressure_strength_m2` | Enabled strength scale × h², otherwise zero, m² |
| `artificial_pressure_reference_separation_m` | Reference separation ratio × h, m |

## Scene document

A scene requires `schema_version`, `name`, `description`, `randomness = "none"`, `[configuration]`
and `[particles]`. `allow_coincident_particles` defaults to `false`; omitted `planes` means no
boundaries. No random construction is supported.

`configuration.source` is `"built_in_defaults"` or `"file"`. The file form requires a non-empty
relative `path`; the built-in form does not accept a path. Optional
`[configuration.overrides.numerical]` and `[configuration.overrides.derived]` use the configuration
fields above, after the base document and before command-line overrides.

`particles.construction = "explicit"` accepts `items`, defaulting to an empty list. Every item
requires a `u64` integer `identity`, a three-component `position_m` and a three-component
`velocity_m_s`. Document identities must be unique and strictly ascending; they need not be
contiguous. At most 4,096 explicit particles are accepted because initial separation is checked
pairwise. This limit does not apply to replayed recorded lattice inputs.

```toml
schema_version = 1
name = "single-particle-example"
description = "One particle above a static floor"
randomness = "none"

[configuration]
source = "built_in_defaults"

[particles]
construction = "explicit"

[[particles.items]]
identity = 7
position_m = [0.0, 1.0, 0.0]
velocity_m_s = [0.0, 0.0, 0.0]

[[planes]]
normal = [0.0, 1.0, 0.0]
offset_m = 0.0
```

For `particles.construction = "lattice"`, every following field is required:

| Field | Contract |
| --- | --- |
| `first_identity` | Initial `u64` identity; subsequent identities increase by one without overflow |
| `origin_m` | Three finite origin coordinates, metres |
| `counts` | Three positive `u32` counts; product at most 1,000,000 |
| `spacing_m` | Positive and bitwise equal to the resolved configuration spacing |
| `endpoint_rule` | `"counted_half_open"`: indices zero through count minus one |
| `order` | `"x_then_y_then_z"`: X outermost, Z fastest |
| `initial_velocity_m_s` | Three finite velocity components, m/s |

Coordinates use binary32 `origin + spacing * index`. Non-finite coordinates, adjacent coordinates
that collapse under rounding, overflowing identities and excessive counts fail before stepping.
The lattice limit bounds construction; it is not a practical brute-force simulation size guarantee.

Each `[[planes]]` entry requires three-component `normal` and finite `offset_m`, in document order.
The normal points into the permitted half-space `normal · position >= offset_m`. Its squared length
must differ from one by at most `1e-6`; it is never silently normalised. At most 32 planes are
accepted. Contradictory geometry and initial particles outside any plane fail validation. Contact
is allowed; particle centres are projected, without a radius offset, friction or restitution.

Explicit initial particles closer than the nominal diameter fail, allowing a rounding margin of
`diameter * (8 * f32::EPSILON)`. `allow_coincident_particles = true` permits exactly coincident
explicit particles for edge-case fixtures; it does not permit other overlaps. This initial-input
check does not introduce a hard separation constraint into later fluid steps.

## Identity and compatibility

Configuration comparison keys include all resolved numerical policies, primary values and derived
values. Scene keys add ordered particle identities/state, planes and randomness. Labels,
descriptions, file paths and field provenance do not decide numerical equivalence. They remain
available for inspection. CLI result records contain the complete effective inputs for replay.

Neighbour discovery is an execution choice (`--backend brute-force`, `serial-grid`,
`threaded-grid` or `cuda`, with `auto` as the resource-selection policy),
not a TOML numerical setting. It leaves configuration and physical input identities unchanged;
backend and run identities distinguish execution. Full-result replay restores the recorded search.

Historical `defaults-v1.toml`, `defaults-v2.toml` and `defaults-v3.toml` remain explicit configurations.
Legacy recorded inputs omit a unit `correction_scale`; the reader supplies one and preserves their
historical comparison identity. Unsupported schema versions and backend arithmetic identities are
rejected. A supported schema is not a promise of bitwise equivalence across compilers or platforms.

The executable's `--help`, the strict readers in `src/config` and `src/scene`, and their integration
tests define the implemented interface. The EPQ numerical and comparison records own the research
experiment; ordinary valid configuration changes do not silently become part of its frozen baseline.
