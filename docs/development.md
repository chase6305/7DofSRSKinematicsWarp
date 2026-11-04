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
