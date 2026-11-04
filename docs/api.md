# API reference

## Python API

```python
import torch
import numpy as np
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

- `get_fk(qpos)` accepts `(7,)` or `(N, 7)` and returns `(N, 4, 4)`.
- `get_all_fk_mat(qpos)` accepts the same inputs and returns `(N, 8, 4, 4)`:
  seven world-frame DH link poses followed by the TCP, computed in one chain
  traversal. Empty and strided batches are supported; each call owns its output.
- `get_ik(target, seed)` accepts `(4, 4)`/`(N, 4, 4)` and a matching
  `(7,)`/`(N, 7)` seed. It returns success `(N,)` and solutions `(N, 1, 7)`.
- With `return_all_solutions=True`, the solution dimension is
  `8 * cfg.num_samples`; invalid slots are zero and valid solutions are first
  when sorting is enabled.
- `get_jacobian()` returns analytical world-frame Jacobians from one joint-chain traversal, including base/tool transforms and joint rotation directions.
- `get_null_space_direction()` and `get_null_space_velocity()` expose the
  one-dimensional redundancy used by fixed-TCP secondary objectives.
- `get_configuration()` returns the shoulder/elbow/wrist branch and arm angle;
  `solve_configuration()` reconstructs that exact point on the IK manifold.
- `solve_configurations(target, configurations, seed=None)` solves a list or
  iterable of exact configurations for one target and shared seed, using CPU
  geometry even when the selected device is CUDA. Returns device tensors:
  boolean `(N,)` success and float32 `(N, 7)` joints in input order. Invalid
  rows are zero and must be masked. Geometry is shared by elbow branch and
  arm-angle decomposition and FK verification are batched.
- Scalar configurations, configuration batches and CPU IK search reuse the most recent local target's geometry, retaining at most two elbow branches. Cache keys use the exact target, DH parameters and link lengths. Seeds, current limits, weights and feasibility checks are evaluated on every call. Fixed-TCP self-motion benefits; moving targets rebuild geometry with a small cache-check overhead. This cache applies to CPU geometry, including exact configuration queries on a CUDA solver; global Warp kernels retain their existing execution path.
- `set_tcp()`/`get_tcp()` update the flange-to-tool transform consistently for
  FK, CPU IK and Warp IK.

For the bundled robot, `KUKAiiwaSolver(device="cuda")` supplies the canonical
iiwa 7 R800 geometry, URDF path and joint limits directly.

## Shared numerical behavior

- CPU `get_ik` retains its continuous search by default. Pass `search_mode="global"` for the same uniform `[-pi, pi)` arm-angle grid used by Warp. Warp currently supports global batch search; requesting continuous mode on Warp raises `ValueError`.
- Both backends map periodic joint representations to the limits nearest the seed. At shoulder/wrist singularities, both outer joints are optimized together using the configured weights and joint limits.
- `solve_configuration(target, configuration, joint_seed=None)` accepts an optional seed for periodic and singular solutions. Branch signs use physical angles, including DH offsets and reversed joint directions.
- Joint weights must be finite and non-negative; rotation directions must be `-1` or `1`; sampling requires an integer of at least two. Pose inputs must be rigid homogeneous transforms. Explicit zero-width limits are retained.
- Empty batches return empty outputs of the documented shapes. Importing the library does not configure application logging.
- Wheels include the URDF and all visual/collision meshes; the solver, viewer and demo resolve installed resources outside the source checkout.

Mixed shoulder/wrist requests in `solve_configurations` are decomposed together
in at most two elbow groups. Output order, per-row validity, periodic mapping,
seed-weighted singular solutions and live joint limits are preserved. This is
a CPU geometry optimization in both repositories, including exact queries on
a CUDA-configured Warp solver; CUDA global search uses a separate candidate kernel.

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
