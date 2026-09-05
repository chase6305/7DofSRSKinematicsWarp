# 7DofSRSKinematicsWarp

Kinematics analytical solution and parallel inverse kinematics solver for SRS 7-DOF robot arm, inspired by the KUKA IIWA architecture.

This repository provides:
- analytical forward and inverse kinematics for 7-DOF serial manipulators
- batched FK and IK acceleration powered by NVIDIA Warp
- CPU fallback with the same Torch batch API
- optional Viser-based visualization

## Environment

- Python: 3.10+
- OS: Linux recommended

## Installation

### 1) Core library only

```bash
pip install -e .
```

### 2) Core + viewer dependencies

```bash
pip install -e .[viewer]
```

### 3) Full development setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Run tests

```bash
pytest
```

### Launch viewer

```bash
srs-viewer --host 127.0.0.1 --port 8080
```

### Run IK interpolation demo

```bash
srs-demo --host 127.0.0.1 --port 8080 --steps 50 --dt 0.05
```

### SRS Viser lab

Install `.[viewer]`, then select one of five experiments (also switchable in the page):

```bash
srs-warp-lab --demo arm-angle --backend warp --device cpu --port 8081
srs-warp-lab --demo branches --backend warp --device cpu --port 8082
srs-warp-lab --demo jacobian --backend warp --device cpu --port 8083
python -m example.viser_trajectory_demo --backend warp --device cpu --port 8084
python -m example.viser_manifold_demo --backend warp --device cpu --port 8085
```

From a checkout, use `python -m example.viser_srs_lab` with the same options. The NumPy repository offers the corresponding `srs-viser-lab` with the same experiment model and controls.

| Mode | Experiment |
|:--|:--|
| `arm-angle` | Fixed-TCP arm-angle sweep with S/E/W markers, shoulder–wrist axis and geometric elbow circle. Infeasible angles retain the last feasible pose and show a message. |
| `branches` | Eight S/E/W branches, availability and minimum joint-limit margin. Display tiles are translated; their local target and arm angle are identical. |
| `jacobian` | Linear velocity ellipsoid and null-space joint centering. Each descent step is corrected back to the exact TCP target on the selected branch. |
| `trajectory` | Fixed-orientation line, circle and figure-eight tracking on the same S/E/W branch, with a gray reference, green FK trace and joint-step/error metrics. |
| `manifold` | Eight fixed-TCP self-motion branches projected onto J1/J3/J5, with infeasible intervals and large joint jumps broken, and a current-configuration marker. |

Trajectory controls select the path, plane, size and joint-speed bound. Path plane offers world XY/XZ/YZ and TCP-local XY, using the TCP orientation at the start. Changing the path, plane or size starts a new reference from the current TCP. Progress slows near constraints and pauses at the last feasible pose if no valid step remains; actual frequency can be lower than the reference frequency. Pause to drag the TCP gizmo. The right plot magnifies displacement along the selected plane axes by six.

All automatic playback modes respect Joint speed bound using direct joint differences per frame, without wrapping away full turns. Up to 12 progressively halved steps let fast arm-angle sweeps and slow Cartesian paths reduce progress before pausing with the last feasible pose and phase intact. Self-motion playback stops at failed steps instead of continuing its sweep to a distant feasible angle. Manual joint, branch and arm-angle controls select poses directly; changing sweep amplitude recenters playback at the current arm angle.

Pausing and resuming preserves the sweep phase, center and direction. Pose, branch, arm-angle, amplitude or experiment changes start a new sweep.

The CLI also accepts `--path`, `--plane`, `--path-size`, `--preset`, `--frequency`, `--joint-speed` and `--amplitude`. Export reproducible playback without installing Viser:

```bash
python -m example.viser_trajectory_demo --backend warp --device cpu --plane tool-xy --path circle --path-size 0.015 --autoplay
python -m example.viser_trajectory_demo --backend warp --device cpu --plane world-xz --path circle --path-size 0.015 --frames 300 --fps 30 --export-csv trajectory.csv
python -m example.viser_manifold_demo --backend warp --device cpu --frames 300 --fps 30 --export-csv self_motion.csv
```

CSV includes the initial pose and each accepted step: simulated time, seven joint angles, actual/target world TCP positions and rotation matrices, pose errors, joint-limit margin, peak joint speed, phase, arm angle and S/E/W branch. Positions are in meters, angles in radians, and matrices are flattened by row. `--fps` accepts 20–240; export uses fixed simulation time steps of `1/fps`, independent of computation time. All five modes export the selected robot. Early pauses retain accepted rows and return exit code 2 with a reason in the JSON report. Existing files are preserved; `--validate-only` and `--export-csv` are mutually exclusive.

Projection axes switches the manifold between J1/J3/J5, shoulder J1/J2/J3, wrist J5/J6/J7 and J2/J4/J6. Curves update only when the target, sampling resolution or projection changes; playback updates the current marker.

Samples / branch sets manifold resolution. Its coordinates are joint angles in radians, rendered at scale 0.22; crossings in the projection need not be identical seven-joint configurations. Finite sampling can miss narrow feasible intervals. Arm angle and Play drive the robot and the current joint-space marker together.

Drag the TCP gizmo to set a target; use joint controls or Pose preset to reset it. Play starts arm-angle sweeps or joint centering. `--autoplay` starts immediately, `--no-mesh` shows only geometry, and `--device cuda` selects an available CUDA device. `--backend numpy --device cpu` uses the batch CPU fallback. The server defaults to `127.0.0.1`.

FK, batched link poses and Jacobians use the selected backend. Exact branch/arm-angle IK and null-space projection in this interactive lab run on the CPU. Paused idle scenes do not recompute or redraw.

Consecutive drag samples from one control are coalesced on insertion. Different controls, reset and mode changes preserve their order so coalescing cannot silently change the TCP target. Rendering and diagnostics share link poses; self-motion reuses the previous frame's Jacobian. Branch IK and link poses are cached by target and arm angle. Automatic stops immediately refresh the final status. The NumPy reference now also accepts batched `get_all_fk_mat` inputs, covered by the cross-repository tests.

The orange elbow circle is a geometric locus; joint limits can exclude parts of it. The ellipsoid shows translational velocity for unit joint-speed norm (display scale 0.25), **with orientation unconstrained**. Actual self-motion uses the full `6×7` Jacobian. Joint centering is local descent, without a global optimality guarantee.

Headless validation requires only core dependencies and checks all five experiments, limits, separate position/rotation residuals and objective descent:

```bash
python -m example.viser_srs_lab --validate-only --backend numpy
python -m example.viser_srs_lab --validate-only --backend warp --device cpu
```

### Run the parallel FK/IK benchmark

The demo builds a batch of reachable poses with FK, solves them together, and
reports throughput and round-trip error. CUDA is selected automatically when
available.

```bash
conda activate hmotion
srs-parallel-demo --device cuda --batch-size 4096 --num-samples 73 --repeats 20
```

For a functional CPU smoke test:

```bash
srs-parallel-demo --device cpu --backend warp --batch-size 64 --num-samples 73 --repeats 10
```

`--backend auto` selects Warp for CUDA and the NumPy fallback for CPU.
`--backend warp` also runs the complete Warp pipeline on CPU, which is useful
for CI, profiling and machines without a visible GPU.

The reference repository's null-space example also has a batched counterpart:

```bash
srs-null-space-demo --device cuda --backend warp --batch-size 4096 --repeats 20
```

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

## Joint validation with the NumPy repository

Keep `7DofSRSKinematics` and `7DofSRSKinematicsWarp` as sibling checkouts to enable `test/test_backend_consistency.py` automatically. Otherwise set `SRS_NUMPY_REPO` to the NumPy checkout. The tests compare all eight branches, FK, global solution selection, and analytical Jacobians. Warp kernels run on CPU and additionally on CUDA when available.

Measure fixed-TCP exact IK, moving-TCP paths and 584-point manifold batches independently, with warmup, median timings and TCP residual checks:

```bash
python tools/benchmark_srs.py --backend warp --device cpu --frames 180 --repeats 15 --output benchmark.json
```

The script needs no viewer dependencies and writes a new JSON file. It measures CPU geometry for exact configuration queries, not GPU global batch IK throughput. NumPy's companion `get_fk` now also accepts `(N, 7)` and returns `(N, 4, 4)`, with empty/strided batches covered by cross-repository tests. NumPy scalar FK retains `(4, 4)` output; this repository retains `(1, 4, 4)` for scalar input.

```bash
SRS_NUMPY_REPO=../7DofSRSKinematics pytest -q
python ../7DofSRSKinematics/tools/compare_backends.py --warp-repo . --samples 40
python -m pip wheel . --no-deps --wheel-dir dist
python tools/check_wheel.py dist/*.whl
```

## Parallel design

For every target pose the analytical SRS solution has three binary branches
(shoulder, elbow and wrist) and one continuous arm angle. The Warp kernel maps
one thread to one `(target, branch, arm-angle sample)` tuple, so a launch covers
`N * 8 * num_samples` candidates without Python loops. FK maps one thread to
one joint configuration.

Target-frame conversion, analytical candidate generation, limit checks and FK
validation stay on the selected Warp device. Nearest-solution reduction and
optional sorting use vectorized Torch operations on the same device. Repeated
calls with an unchanged batch size reuse the large Warp work buffers.

## Repository Layout

```text
.
├── srs_solver.py
├── warp_kernel.py
├── viser_viewer.py
├── example/
│   ├── demo.py
│   ├── parallel_demo.py
│   └── parallel_null_space_demo.py
├── test/
│   ├── test_batch_api.py
│   ├── test_srs_solver.py
│   └── test_warp_kernel.py
├── urdf/
│   ├── lbr_iiwa7_r800.urdf
│   └── meshes/
└── pyproject.toml
```
