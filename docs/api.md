# API reference

## Python API

```python
import torch
from srs_solver import SRSSolver, SRSSolverCfg

cfg = SRSSolverCfg()
solver = SRSSolver(cfg, num_envs=1, device="cpu")

q_known = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]])
target_xpos = solver.get_fk(q_known)  # (batch, 4, 4)
q_seed = torch.zeros_like(q_known)

success, qpos = solver.get_ik(target_xpos, q_seed, return_all_solutions=False)
print(success, qpos)
```

Batch conventions:

- Tensor results use float32 on the solver device; success flags use bool.
  The application's Torch default dtype or autocast scope does not change these
  types or reduce the precision of pose validation and null-space projection.
  CPU IK and exact queries check FK after converting joints to float32. A
  float64 candidate that loses the required pose accuracy during conversion
  is rejected, which can matter with large multi-turn joint coordinates.
- `get_fk(qpos)` accepts `(7,)` or `(N, 7)` and returns `(N, 4, 4)`.
- `get_all_fk_mat(qpos)` accepts the same inputs and returns `(N, 8, 4, 4)`:
  seven world-frame DH link poses followed by the TCP, computed in one chain
  traversal. Empty and strided batches are supported; each call owns its output.
- `get_ik(target, seed)` accepts `(4, 4)`/`(N, 4, 4)` and a matching
  `(7,)`/`(N, 7)` seed. It returns success `(N,)` and solutions `(N, 1, 7)`.
- With `return_all_solutions=True`, the solution dimension is
  `8 * cfg.num_samples`; invalid slots are zero and valid solutions are first
  when sorting is enabled. Success remains per target; no separate
  candidate-validity mask is returned. This option is for the sampled search;
  `interval` returns one solution per target and rejects this option.
- `get_jacobian(q)` returns `(6, 7)` for `(7,)` input or `(N, 6, 7)` for a
  batch, in world coordinates with linear velocity first. Base/tool transforms
  and joint rotation directions are included.
- `get_null_space_direction()` returns one deterministic unit null direction.
  `get_null_space_velocity()` projects into the full Jacobian null space through
  orthonormal SVD vectors, retaining extra directions at singular poses. Its rank
  cutoff follows [Torch’s default pseudoinverse
  threshold](https://docs.pytorch.org/docs/stable/generated/torch.linalg.pinv.html).
- `get_configuration()` returns the shoulder/elbow/wrist branch and arm angle;
  `solve_configuration()` reconstructs that exact point on the IK manifold.
- `solve_configurations(target, configurations, seed=None)` solves a list or
  iterable of exact configurations for one target and shared seed, using CPU
  geometry even when the selected device is CUDA. Returns device tensors:
  boolean `(N,)` success and float32 `(N, 7)` joints in input order. Invalid
  rows are zero and must be masked. Geometry is shared by elbow branch and
  arm-angle decomposition and FK verification are batched.
- Exact configuration queries and CPU IK reuse the most recent target geometry;
  seeds, limits and weights remain live on every call.
- `set_tcp()`/`get_tcp()` update the flange-to-tool transform consistently for
  FK, CPU IK and Warp IK.

For the bundled robot, `KUKAiiwaSolver(device="cuda")` supplies the canonical
iiwa 7 R800 geometry, URDF path and joint limits directly.

## Search modes

| Mode | Execution and search policy |
| --- | --- |
| `continuous` | NumPy CPU default: seed branch and local offsets, uniform grid, then interval fallback. |
| `global` | Native Warp default: uniform arm-angle samples over eight branches on the selected device. Also available in NumPy. |
| `interval` | Float64 CPU geometry: joint-limit boundary events and cell midpoints, on either backend. |

For narrow feasible ranges, use:

```python
success, joints = solver.get_ik(target_xpos, q_seed, search_mode="interval")
```

This explicit mode also works on a CUDA solver, but transfers inputs to CPU for
geometry and returns float32 tensors on the solver device. It favors narrow-limit
reliability over GPU throughput, and does not depend on `cfg.num_samples`.
If boundary search fails, the limit-box midpoint and projected seed are checked
by FK to handle selected degenerate cases. Native global batching and its
candidate shapes are unchanged. Continuous mode
still requires the NumPy backend. None of the modes guarantees a continuous
global optimum or implements the papers' interval-based avoidance policy.

## Shared numerical behavior

- Both backends map periodic joint representations to the limits nearest the seed. At shoulder/wrist
  singularities, both outer joints are optimized together using the configured weights and joint
  limits.
- `solve_configuration(target, configuration, joint_seed=None)` accepts an optional seed for
  periodic and singular solutions. Branch signs use physical angles, including DH offsets and
  reversed joint directions.
- Joint weights must be finite and non-negative; rotation directions must be `-1` or `1`; sampling
  requires an integer of at least two. Pose inputs must be rigid homogeneous transforms. Explicit
  zero-width limits are retained.
- Workspace checks allow float32 rounding at full extension; returned solutions still pass FK
  residual checks.
- Empty batches return empty outputs of the documented shapes. Importing the library does not
  configure application logging.
- Reversed NumPy slices such as `q[::-1]` and `q[:, ::-1]` are accepted. Views with negative strides
  are copied before tensor conversion, including joint states, IK targets/seeds and TCP transforms.
- Wheels include the URDF and all visual/collision meshes; the solver, viewer and demo resolve
  installed resources outside the source checkout.

Joint-distance weights are normalized before ranking: multiplying all weights
by the same positive finite factor does not change the selected solution.
Zero weights retain deterministic input ordering for tied candidates.

CUDA FK, link poses, Jacobians and global IK execute on the caller's current
PyTorch stream. Warp and Torch share that stream for zero-copy inputs and
outputs. Sequential calls on different streams join through the solver's Warp
stream so its cached workspace cannot race the previous call. A solver instance
is not intended for simultaneous calls from multiple Python threads; use one
instance per concurrent worker. Returned public outputs own their storage and
remain valid after subsequent calls, including calls with another batch size.
