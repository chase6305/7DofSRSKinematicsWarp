# Development and benchmarks

## Joint validation

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

Input checks, target-frame conversion, analytical candidate generation, limit
checks, FK verification and distance scoring stay on the selected Warp device.
Two elbow geometries are prepared per target, then shared by its arm-angle and
shoulder/wrist candidates. Torch performs nearest-solution reduction and stable
sorting on that same device. Repeated calls with an unchanged batch size reuse
the large Warp work buffers; geometric values are recomputed on each call.

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

## Exact-configuration batch optimization

Eight S/E/W branches share only two elbow geometries. Batch requests now
decompose all matching shoulder/wrist rows together, reducing up to eight
decomposition calls to two. Shuffled input order, zero/π shoulder and wrist
singularities, periodic limits, seeds and reversed joints are covered by tests.

A local CPU comparison against the previous implementation reused the same
target, seed and configuration objects. Nine interleaved repeats of 30 warmed
calls produced these medians; output masks and joint arrays were exactly equal:

| Exact requests | Before | After | Speedup |
|:--|--:|--:|--:|
| Eight-branch gallery (8) | 0.344 ms | 0.153 ms | 2.25× |
| Self-motion sampling (584) | 0.749 ms | 0.539 ms | 1.39× |

These timings use Python 3.10 and NumPy 2.2.6, and exclude request construction,
the viewer, and WebGL. Warp's exact-configuration API also uses CPU geometry;
the figures do not measure GPU global IK throughput. Hardware and workload
affect the timings.

Run the companion benchmark from the checkout to measure scalar fixed/moving
TCP queries, the eight-branch batch, and the 584-point manifold batch. The
benchmark includes adapter/request construction and verifies TCP residuals:

```bash
python tools/benchmark_srs.py --backend warp --device cpu --frames 180 --repeats 15
```

## CUDA candidate scoring

Global Warp IK now scores one candidate per thread into a reused scalar buffer.
It avoids materializing the two `(N, 8 * samples, 7)` subtraction/multiplication
arrays. Torch still performs the stable sort or first-minimum reduction;
zero-weight ties retain candidate order and invalid rows remain zero.

A local comparison isolated Torch scoring versus the fused Warp score kernel,
with the same stream handling and normalized weights in both versions. Each
batch used 73 samples, seed offsets of 0.02 rad, three warmup calls and 15
interleaved, synchronized repeats. The selected joints were bit-identical for
all targets:

| Targets | Torch scoring | Fused scoring | Torch peak temporary allocation, before → after |
|--:|--:|--:|--:|
| 64 | 0.645 ms | 0.623 ms | 2.03 → 0.04 MiB |
| 1024 | 0.679 ms | 0.634 ms | 32.57 → 0.64 MiB |
| 4096 | 1.029 ms | 0.826 ms | 130.28 → 2.57 MiB |

The reused Warp score buffer adds 9.13 MiB at 4096 targets, increasing the Warp
workspace from 73.25 to 82.38 MiB. The table's allocation figures count Torch
allocations only; they exclude Warp buffers and cached allocator reservations.
These measurements use Python 3.10, Torch 2.13.0+cu130, Warp 1.16.0 and an RTX
5090 D v2. They include input validation and result selection, exclude kernel
compilation and the viewer, and vary with hardware and load.

Run the current global CUDA workload with:

```bash
srs-parallel-demo --device cuda --batch-size 4096 --num-samples 73 --repeats 20
```

Tests cover uniform, unequal and zero weights; all/nearest results; failed rows;
workspace resize; retained outputs; live weight edits; and launches on two
non-default PyTorch streams. CPU kernels run in CI; CUDA checks activate when a
device is available.

## CPU global search

Global search now evaluates the branch-major arm-angle grid in the same two
elbow groups as exact configuration batches. Continuous local search keeps its
existing seed/branch priority. Global candidates retain their original order,
so zero-weight ties and all-solution ordering stay deterministic.

A local fixed-target comparison (Python 3.10, NumPy 2.2.6; nine interleaved
repeats of 30 warmed calls, seed offset 0.2 rad) measured:

| Arm-angle samples | Before | Grouped | Speedup |
|--:|--:|--:|--:|
| 73 | 0.927 ms | 0.653 ms | 1.42× |
| 361 | 1.319 ms | 1.148 ms | 1.15× |

Across 240 additional cases shared between the NumPy reference and CPU fallback,
before/after results matched, including unequal/zero weights, shoulder/wrist
singularities and all-solution requests. These are CPU timings, excluding the
viewer; finite sampling still does not guarantee finding every feasible interval.
The two repositories use different default limits, so compare before/after
within each row rather than treating the tables as a backend ranking.

Measure the current implementations with the cross-backend tool:

```bash
python ../7DofSRSKinematics/tools/compare_backends.py --warp-repo . --samples 40 --num-samples 73
```

## Null-space projection near singular poses

`get_null_space_velocity` projects preferred velocities through the orthonormal
null-space vectors from a full SVD. This retains extra null directions at
singular poses and avoids cancellation from constructing `I - pinv(J) @ J`.
The rank cutoff matches Torch's default pseudoinverse cutoff.

A local comparison used fixed float32 Jacobians, random preferred velocities,
three warmups and 15 interleaved repeats. The reference uses float64 SVD on the
same Jacobians and the same float32 rank cutoff:

| Device / batch | Maximum task residual before → after | Maximum component error versus float64 before → after |
|:--|--:|--:|
| CPU / 512 | 2.70e-6 → 2.65e-7 | 4.09e-6 → 3.08e-6 rad/s |
| CUDA / 4096 | 4.59e-6 → 2.41e-7 | 2.02e-5 → 4.12e-6 rad/s |

The task residual is `max(norm(J @ velocity))` and mixes linear/angular velocity
units. These measurements exclude Jacobian generation. Runtime was 2.08 →
1.98 ms on CPU and 1.15 → 1.14 ms on CUDA; the numerical improvement is the main
benefit. The environment matches the CUDA scoring benchmark above. Tests check
annihilated task velocity, idempotence, non-increasing velocity norm, exact and
near-straight singular configurations, input preservation and empty batches.

## Shared device geometry and input validation

Global IK prepares two elbow geometries per target instead of repeating that
work for every candidate. The preparation launch also converts world targets to
the local arm frame, retaining the previous launch count. The public direct
`compute_ik_kernel` signature remains available. Tests compare every candidate
with that path, including singular poses, reversed joints, offsets, base/tool
frames and changing targets/seeds.

Finite/rigid target checks and seed checks now use one Warp pass and one host
status read. They retain error priority and the 1e-6 rigid-transform tolerance;
rejected batches do not poison subsequent calls. An additional stress run
accepted 262,144 FK/external rigid transforms across CPU and CUDA. Unindexed
`cuda` follows Torch's current device when creating the solver, then keeps that
concrete device for subsequent calls. Indexed CPU aliases normalize to `cpu`.

A local comparison against `8472c8b` used 73 arm-angle samples, seed offsets of
0.02 rad, five warmups and 21 interleaved repeats. All candidate masks and selected
joint tensors were bit-identical in the measured batches:

| Device / targets | Before | After |
|:--|--:|--:|
| Warp CPU / 64 | 13.491 ms | 10.181 ms |
| CUDA / 1 | 0.601 ms | 0.249 ms |
| CUDA / 64 | 0.594 ms | 0.245 ms |
| CUDA / 4096 | 0.962 ms | 0.480 ms |

These are complete `get_ik` timings, including validation and result selection,
with CUDA synchronization, excluding compilation and the viewer. The environment
matches the CUDA scoring table above. Geometry storage adds 1.75 MiB at 4096
targets (82.38 → 84.13 MiB of Warp workspace). Hardware/load affect timings;
see the [raw measurements](benchmarks/global-ik.json). The parallel demo command
above measures the current implementation.

Distance scoring also handles finite seeds whose squared norms overflow
float32. That exceptional path computes the norm in float64 and bounds it to a
finite score, so invalid candidates remain behind valid ones. Ordinary control
loop inputs retain float32 scoring. Regression tests cover nearest/all results,
stable ties and zero-weight components at seeds up to 3e38.

FK also checks finite joints in its existing device launch. The public
kernel marks nonfinite rows unsuccessful and writes zero poses; `get_fk`
reports the same input error after reading those flags, avoiding a separate
Torch finite-check pass. Strided batches, recovery after rejected rows and
non-default CUDA streams are covered by tests.

The native FK kernel unrolls its seven-joint chain so Warp Tape retains the
intermediate poses required for reverse mode. A dynamic loop previously
produced incorrect derivatives even though forward poses were correct. CPU/CUDA
tests compare Tape position gradients with the analytical Jacobian, including
reversed joint axes and a translated TCP. Generic native FK joint counts retain
their previous forward computation.

CPU compatibility checks include Warp 1.5.0 / NumPy 1.24.4 in CI. Kernel control
flow uses branches supported by that older compiler; current Warp releases also
run the full CPU/CUDA suite locally.

## Viewer startup and branch gallery

The primary robot parses the URDF once. Gallery robots keep independent joint
states/transform graphs while sharing source meshes and link-local GLB bytes.
All eight instances remain lazy and are reused after mode switches. Integration
checks compare cached GLBs byte-for-byte with normal Viser exports.

The shared viewer implementation reduced the first gallery switch from 489 ms
to 230 ms in a local NumPy-backend comparison against companion commit
`8174d96` (seven interleaved trials after one warmup). It includes IK and server
scene construction, excluding initial app construction, browser, network and
WebGL. See [environment and raw measurements](benchmarks/viewer.json).

Measure the current viewer with its optional dependencies installed:

```bash
python tools/benchmark_viewer.py --backend warp --device cpu --repeats 7 --output viewer-benchmark.json
```

The tool needs no browser. It measures startup and first/repeated gallery
opening separately, verifies robot reuse and TCP residuals, and writes a new
JSON file with dependency versions and samples.
