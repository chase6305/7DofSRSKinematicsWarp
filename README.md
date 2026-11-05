# 7DofSRSKinematicsWarp

English | [简体中文](README_ZH.md) · [NumPy version](https://github.com/chase6305/7DofSRSKinematics)

Analytical kinematics for 7-DoF S-R-S arms, with GPU batching through NVIDIA
Warp and PyTorch. Includes a KUKA iiwa 7 R800 example.

- Analytical FK / IK, eight S/E/W branches and arm-angle control.
- Batched link poses, Jacobians and null-space motion.
- Interactive demos and CSV export.

## Demos

### Fixed-TCP self-motion · ±90° arm-angle sweep

![Fixed TCP with a large elbow sweep](docs/media/self-motion.gif)

### Figure-eight TCP tracking · 180 × 90 mm

![Figure-eight reference and actual TCP trace](docs/media/trajectory.gif)

Two ten-second recordings of the running viewer.

## Quick start

Python 3.10+:

```bash
python -m pip install -e '.[viewer]'
srs-warp-lab --demo arm-angle --amplitude 90 --joint-speed 2 --autoplay
# Or run the figure-eight demo:
srs-warp-lab --demo trajectory --path-size 0.09 --autoplay
```

Open <http://127.0.0.1:8081>. Select an experiment and toggle **Play**.
The sidebar also offers eight branches, a Jacobian ellipsoid and the self-motion
manifold. [Controls and CSV export](docs/demos.md).

## Python API

```python
import torch
from srs_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver(device="cpu")  # use "cuda" for GPU batches
q = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]])
target = solver.get_fk(q)
ok, joints = solver.get_ik(target, torch.zeros_like(q))
print(joints[ok, 0])  # successful rows, radians
```

[API reference](docs/api.md) covers batch shapes, frames and limits.

## Benchmark

Global IK on an RTX 5090 D v2 with Warp 1.16 and 73 arm-angle samples:

| Batch size | Median latency |
| ---: | ---: |
| 64 | 0.245 ms |
| 4096 | 0.480 ms |

Warmed complete calls, including validation and selection; compilation excluded.
[Raw results](docs/benchmarks/global-ik.json).

```bash
srs-parallel-demo --device cuda --batch-size 4096 --num-samples 73
```

## References

- Shimizu et al. (2008). [Analytical Inverse Kinematic Computation for 7-DOF
  Redundant Manipulators With Joint Limits and Its Application to Redundancy
  Resolution](https://doi.org/10.1109/TRO.2008.2003266).
  *IEEE Transactions on Robotics*, 24(5), 1131–1142.
- Faria et al. (2018). [Position-based kinematics for 7-DoF serial manipulators
  with global configuration control, joint limit and singularity
  avoidance](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).
  *Mechanism and Machine Theory*, 121, 317–334.

The implementation follows the arm-angle formulation, with sampled arm-angle
search rather than complete analytical feasible-interval construction.
See [implementation differences](docs/paper-comparison.md).
