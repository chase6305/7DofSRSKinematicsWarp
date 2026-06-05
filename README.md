# 7DofSRSKinematicsWarp

Kinematics analytical solution and parallel inverse kinematics solver for SRS 7-DOF robot arm, inspired by the KUKA IIWA architecture.

This repository provides:
- analytical forward and inverse kinematics for 7-DOF serial manipulators
- batched IK acceleration powered by NVIDIA Warp
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

## Python API

```python
import torch
import numpy as np
from srs_solver import SRSSolver, SRSSolverCfg

cfg = SRSSolverCfg()
solver = SRSSolver(cfg, num_envs=1, device="cpu")

q_seed = torch.zeros((1, 7), dtype=torch.float32)
target_pose = np.eye(4, dtype=np.float32)
target_xpos = torch.from_numpy(target_pose).unsqueeze(0)

success, qpos = solver.get_ik(target_xpos, q_seed, return_all_solutions=False)
print(success, qpos)
```

## Repository Layout

```text
.
├── srs_solver.py
├── warp_kernel.py
├── viser_viewer.py
├── example/
│   └── demo.py
├── test/
│   └── test_srs_solver.py
├── urdf/
│   ├── lbr_iiwa7_r800.urdf
│   └── meshes/
└── pyproject.toml
```
