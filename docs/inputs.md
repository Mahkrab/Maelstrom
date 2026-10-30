# Scenes, geometry and configuration

Maelstrom accepts strict TOML inputs with `schema_version = 1` or `2`. Unknown fields and unsupported
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

Each new package is `<scenes-root>/water/scene.toml`, `config.toml` and `render.toml`. The scene document
holds particle/lattice geometry and ordered static planes. Its configuration reference
is scene-relative, so moving the package does not depend on the shell's working
directory. Creation copies effective primary numerical values, not inspection-only
derived diagnostics. Creation writes schema 2, clears numerical overlays and removes duplicate lattice
spacing after validation. It copies the source's saved run limit, or writes a visible one-second limit.
Presentation settings are discovered beside the scene without a launch flag. There is no mesh or
arbitrary external geometry format.

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
The preferences schema permits only `schema_version = 1`, `numerical_mode`, `backend`, `threads`,
`device` and `block_size`. Explicit execution selection takes precedence; these
backend preferences do not alter physical inputs. `numerical_mode` selects `fast` (the default)
or `reference` (`scene` preserves a saved custom schedule); its scheduling preset overrides scene/configuration scheduling fields,
while explicit `--set` values take precedence. For example:

```toml
schema_version = 1
backend = "threaded-grid"
threads = 4
```

Storage environment overrides must be absolute. Relative explicit CLI paths are
resolved against the invocation directory. Invalid/relative XDG roots use the normal
HOME fallback. Unknown preference fields and schema versions fail rather than changing
physical parameters such as geometry, density and gravity.

## Configuration resolution

A configuration document requires `schema_version`, `name` and `description`. Its optional
`[numerical]` table supplies primary values; its optional `[derived]` table asserts calculated values.
An omitted primary field inherits the earlier layer. Resolution applies:

1. Built-in defaults, currently equivalent numerically to `config/defaults-v4.toml`.
2. The selected configuration document.
3. The scene's `configuration.overrides` overlay.
4. Command-line `--set FIELD=VALUE` overrides; `--refine-timestep N` also selects refinement.

Scene and numerical configuration versions must match, including an explicit `--config` replacement.
For schema 2, one refinement factor is selected through these same layers, then applied once after
primary values resolve. A CLI factor of one resets saved refinement. Every supplied derived assertion
must match the final derived values; invalid supplied primaries cannot be hidden by later overrides.
Schema 1 retains its original per-layer refinement and per-layer derived checks. Existing scenes
can be migrated without changing their effective physics using `scenes create NAME --from SOURCE`.
Migration preserves explicitly supplied assertions as final effective assertions and reports replacements;
it does not add derived assertions to an ordinary scene that had none.

An external scene's configuration path is relative to that scene's directory. `--config FILE`
replaces the selected base document and retains the scene overlay. Bundled scenes and their base
configuration are embedded in the executable, so named scenes also work outside the checkout.
The build discovers the top-level `scenes/*.toml` and `config/*.toml` files in sorted order;
scene selectors come from filenames without `.toml`. Adding or removing a shipped scene
requires a rebuild, not a Rust registry edit. Filenames use ASCII letters, digits, hyphens
and underscores, with stems of at most 120 characters. Each scene retains its own relative
configuration reference. Personal scenes belong in the editable library and need no rebuild.
Environment variables do not select numerical inputs. `inspect --format json` exposes resolved values,
derived quantities, field provenance, refinement selection and technical comparison keys. Source
documents carry portable labels and SHA-256 identities. Inspection also resolves saved run and render
settings; `--steps`, `--duration`, `--render-config`, dimension, stride and fit flags preview overrides.
`validate` checks presentation inputs without initialising graphics. Headless `run` ignores rendering.

In schema 1, each layer applies primary overrides, then any refinement, derives quantities and checks
its own assertions. In schema 2, refinement operates on the final selected primary values.
Refinement divides the effective timestep by N and the complete position
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

Presentation choices live in the scene's `render.toml`, independently of these numerical values.
[The viewer guide](viewer.md#particle-colours) covers fixed and kinetic-energy colours, automatic
camera fitting and explicitly saving a camera. Viewing and compilation discover the same file;
headless execution does not parse it.

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
| `xsph_viscosity_enabled` | `false` | Boolean; enables a velocity-smoothing pass after reconstruction |
| `xsph_viscosity_coefficient` | `0.01` | Dimensionless strength, finite 0..=1; zero or disabled bypasses smoothing |
| `vorticity_confinement_enabled` | `false` | Boolean; restores rotational motion before XSPH |
| `vorticity_confinement_strength_m_s` | `0.01` | Finite non-negative strength, metres per second; zero or disabled bypasses confinement |

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

## Fluid motion controls

XSPH reduces local velocity differences after position correction and velocity reconstruction. It
reads the complete velocity snapshot after optional confinement and applies one ordered Poly6/rest-density
weighted correction per timestep. It changes accepted velocities first, then subsequent motion;
it does not modify that timestep's projected positions. It works with serial, threaded and CUDA
execution and remains disabled for existing configurations.

Save this selection in the scene's numerical configuration, or make a temporary CLI override.
These commands start fresh simulations, headlessly and in a live window respectively:

```sh
maelstrom run water --steps 120 --set xsph_viscosity_enabled=true --set xsph_viscosity_coefficient=0.01
maelstrom view water --set xsph_viscosity_enabled=true --set xsph_viscosity_coefficient=0.01
```

Start with a small coefficient. This is numerical smoothing, not a calibrated material viscosity in
Pa s, and compressed neighbourhoods can oversmooth. The coefficient is per timestep: refinement
applies it more often over a given simulated duration, so reassess it when changing the timestep.
Disabled or zero strength leaves the previous calculation unchanged. Active strength and the
smoothing revision are included in comparison identities and recorded inputs; reruns retain them.
Vorticity confinement restores rotational motion after velocity reconstruction and before XSPH.
It estimates curl and the direction of increasing curl magnitude from complete snapshots using
rest-volume Spiky gradients. Its acceleration is integrated with the timestep; uniform motion and
zero curl or zero magnitude gradient receive no correction. Serial, threaded and CUDA support it.

Save these fields in `[numerical]` in `config.toml` to keep the selection for future launches:

```toml
vorticity_confinement_enabled = true
vorticity_confinement_strength_m_s = 0.01
xsph_viscosity_enabled = true
xsph_viscosity_coefficient = 0.01
```

For a temporary selection, this starts a fresh simulation:

```sh
maelstrom view water --set vorticity_confinement_enabled=true --set vorticity_confinement_strength_m_s=0.01
```

Confinement strength is an energy-restoration control in m/s, not a physical viscosity or a guaranteed
stable range. Start small: large values can inject excessive energy. Disabled or zero strength leaves
the previous calculation unchanged. Strength and the confinement revision are recorded for reruns
and comparisons. Changing timestep, resolution or strength needs renewed assessment; there is no
promise of identical trajectories after refinement.

Spacing, support, gravity, timestep, solver iterations, relaxation, correction response and artificial
pressure affect dynamics. Artificial pressure supplies a repulsive correction, rather than a calibrated
surface-tension material model. Physical rest density sets mass and reported physical density; it
also affects energy colours, while equal-mass constraint dynamics and rest-volume velocity-effect weights remain
unchanged. The CFL factor changes warnings, without changing trajectories or rejecting a timestep.

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

Schema 2 accepts an optional `[run]` with exactly one of `steps` or `duration_s`. Steps are a
non-negative integer; duration is finite, non-negative simulated seconds rounded up using the resolved
timestep. Empty or conflicting selections fail. `run`, `view` and `compile` use the saved limit unless
an explicit `--steps` or `--duration` replaces it. Without either a saved or explicit limit, execution
asks for one. Checkpoints, measurement, pacing overrides and output paths remain command options.

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

For `particles.construction = "lattice"`, the following fields apply:

| Field | Contract |
| --- | --- |
| `first_identity` | Required in schema 1; optional in schema 2. Defaults to zero for a standalone lattice, or follows the preceding group; identities increase by one without overflow |
| `origin_m` | Three finite origin coordinates, metres |
| `counts` | Three positive `u32` counts; product at most 1,000,000 |
| `spacing_m` | Required in schema 1; optional in schema 2. Omission inherits numerical spacing; an explicit value must be positive and bitwise equal to it |
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

## Multiple initial groups and a container

Start from an editable two-block example rather than writing plane normals:

```sh
maelstrom scenes create collision --from builtin:two-blocks
maelstrom inspect collision --backend serial-grid
maelstrom view collision
```

Viewing starts a fresh simulation. The copy keeps the example's groups, shared numerical configuration
and saved run limit; `inspect collision` shows their effective values. Its
[scene document](../scenes/two-blocks/scene.toml) shows the complete format. The renderer fits the finite
container by default.

Schema 2 accepts `particles.construction = "groups"` followed by `[[particles.groups]]` tables.
Each table has a unique, case-sensitive, non-blank `name` and the same `construction = "lattice"`
or `"explicit"` fields described above. Nested groups and per-group material settings are rejected.
There are at most 256 groups and one million particles across them; an empty collection is valid.

Groups retain document order. A lattice without `first_identity` starts at zero if no preceding
particle exists, otherwise one after the last identity. Explicit particles keep their declared
identities; an explicit lattice start is also retained. Every non-empty group must begin above the
preceding group's last identity. Empty explicit groups consume no identities. The solver receives
one ascending particle vector; group names do not create separate fluids or change comparison keys.

Grouped geometry uses a spatial overlap check for explicit sources and between groups.
Within a generated lattice, the existing ordered-axis checks reject coordinate collapse; nominal
spacing is established by construction. Rounded decimal coordinates are not compared again against
the nominal diameter within that same lattice. No coordinates or spacing values are adjusted. Grouped explicit
sources therefore share the one-million total limit rather than the standalone 4,096-particle cap.
Exactly coincident points are admitted only with `allow_coincident_particles = true`; other overlaps
still fail. Cell coordinates, calculated as world position divided by numerical spacing, must fit
signed 64-bit integers. These are construction limits, not a backend capacity guarantee.

An optional `[container]` describes an axis-aligned enclosing box in metres. Supply either
`min_m` and `max_m`, or lower-corner `origin_m` and positive `dimensions_m`. Do not mix forms.
Bounds must remain finite and strictly increasing after binary32 calculation. For example:

```toml
[container]
min_m = [0.0, 0.0, 0.0]
max_m = [0.8, 0.7, 0.6]
```

The box expands into inward-facing planes in X-minimum, X-maximum, Y-minimum, Y-maximum,
Z-minimum, Z-maximum order. Explicit `[[planes]]` follow in their document order and intersect
that domain. The 32-plane limit includes the six generated planes. Bounds constrain particle
centres under the existing projection model; a box is not a solid obstacle or mesh collider.
Schema 1 keeps its existing single-source and explicit-plane format.

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
