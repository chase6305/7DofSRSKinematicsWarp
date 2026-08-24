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
- `get_ik(target, seed)` accepts `(4, 4)`/`(N, 4, 4)` and a matching
  `(7,)`/`(N, 7)` seed. It returns success `(N,)` and solutions `(N, 1, 7)`.
- With `return_all_solutions=True`, the solution dimension is
  `8 * cfg.num_samples`; invalid slots are zero and valid solutions are first
  when sorting is enabled.
- `get_jacobian()` batches the 14 central FK differences in one call.
- `get_null_space_direction()` and `get_null_space_velocity()` expose the
  one-dimensional redundancy used by fixed-TCP secondary objectives.
- `get_configuration()` returns the shoulder/elbow/wrist branch and arm angle;
  `solve_configuration()` reconstructs that exact point on the IK manifold.
- `set_tcp()`/`get_tcp()` update the flange-to-tool transform consistently for
  FK, CPU IK and Warp IK.

For the bundled robot, `KUKAiiwaSolver(device="cuda")` supplies the canonical
iiwa 7 R800 geometry, URDF path and joint limits directly.

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
