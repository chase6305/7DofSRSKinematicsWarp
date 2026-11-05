# Development and benchmarks

## Verification

```bash
SRS_NUMPY_REPO=../7DofSRSKinematics python -m pytest -q
srs-warp-lab --validate-only --backend warp --device cpu
python -m pip wheel . --no-deps --wheel-dir dist
python tools/check_wheel.py dist/*.whl
```

Sibling NumPy checkouts enable cross-repository tests automatically; otherwise
set `SRS_NUMPY_REPO`. They compare FK, Jacobians, all eight branches and global
solution selection. Warp tests run on CPU and additionally on CUDA when
available. CI also checks Warp 1.5.0 with Torch 2.1.2 and NumPy 1.24.4.

## Solver design

Global IK maps one Warp thread to each `(target, branch, arm-angle sample)`.
Two elbow geometries per target are shared by its eight S/E/W branches and all
arm-angle candidates. Input validation, frame conversion, candidate generation,
limit/FK checks and scalar distance scoring stay on the selected device. Torch
selects the nearest candidate or performs a stable sort on that device.

Repeated calls reuse Warp workspaces while returning independent result tensors.
Kernels follow the current PyTorch CUDA stream. Equal scores retain candidate
order; failed rows are zero and must be masked. Finite arm-angle sampling can
miss feasible intervals. The exact branch/arm-angle API uses CPU geometry even
when the selected device is CUDA; see [API contracts](api.md) for shapes and
scalar conventions.

Explicit `interval` search uses float64 CPU geometry to check analytical
joint-limit boundaries and cell midpoints. It preserves CPU input precision
and avoids native kernel launches; only the result moves to the selected
device. Boundary events reuse up to two geometry/limit-specific arrays, while
seeds, weights and FK residual checks remain live. NumPy continuous mode uses
this search only after its local and uniform queries fail.

FK checks finite inputs in its existing launch. Its seven-joint chain retains
intermediates for Warp Tape gradients. Null-space velocity uses a full SVD to
preserve extra null directions at singular poses. Modern Warp versions compile
kernels separately; Warp 1.5 retains the compatible module registration path.

The NumPy CPU fallback reuses one DH matrix buffer for batches of at least 256
robots and preallocates link poses and Jacobian storage; small batches retain
eager DH evaluation.

Repeated 584-request manifold batches now reuse frozen configurations and
branch/angle metadata. Complete warmed calls changed from 1.398 to 0.591 ms
(about 58% lower latency) against the preceding implementation. The comparison
uses CPU affinity 30, one BLAS/OpenMP thread and ABBA process order, with 30
raw trials per revision. [Request-reuse data](benchmarks/request-reuse.json)
includes the commands, revisions and all measured workloads. These are CPU
exact-query timings, including request construction and output conversion.

## Reproduce benchmarks

```bash
srs-parallel-demo --device cuda --backend warp --batch-size 4096 --num-samples 73 --repeats 20
python tools/benchmark_startup.py --device cuda --batch-size 64 --output startup-benchmark.json
python tools/benchmark_srs.py --backend warp --device cpu --frames 180 --repeats 15 --output exact-benchmark.json
python tools/benchmark_viewer.py --backend warp --device cpu --repeats 7 --output viewer-benchmark.json
python ../7DofSRSKinematics/tools/compare_backends.py --warp-repo . --samples 40 --device cuda
```

Use `--device cpu` for machines without CUDA. The parallel demo measures global
batch IK; `benchmark_srs.py` measures fixed/moving TCP and exact-configuration
batches. The viewer tool needs the viewer extra but no browser. Startup uses a
temporary Warp cache and disables driver caching only in its own process.
Benchmarks report residuals alongside timing and separate compilation from
warmed calls. Existing JSON output files are preserved.

Recorded local comparisons on RTX 5090 D v2, Python 3.10, Torch 2.13.0+cu130,
Warp 1.16.0 and NumPy 2.2.6:

| Workload | Baseline | Optimized |
| :-- | --: | --: |
| Global IK, 64 CUDA targets | 0.594 ms | 0.245 ms |
| Global IK, 4096 CUDA targets | 0.962 ms | 0.480 ms |
| Constructor + first FK, CUDA | 20.371 s | 2.231 s |
| Constructor through first IK, CUDA | 20.421 s | 8.738 s |

[Global IK data](benchmarks/global-ik.json) compares against `8472c8b`, with
73 arm-angle samples, validation, selection and CUDA synchronization included.
[Startup data](benchmarks/startup.json) compares against `b88dec1`, with 64
targets in separate processes and empty caches. Imports and the viewer are
excluded. Hardware, compiler and cache behavior affect timings.

The shared viewer also reduced first gallery opening from 545 to 254 ms in the
NumPy-backend comparison recorded in [viewer data](benchmarks/viewer.json).
That measures IK and server construction, excluding initial app construction,
network, browser and WebGL; it is separate from CUDA solver throughput.

For 4096 CPU fallback robots, complete FK calls changed from 1.861 to 1.366 ms
and peak tracked allocations from 5.51 to 2.26 MB against `aff1574`.
[Batch-chain data](benchmarks/batch-chain.json) includes link poses, Jacobians,
raw trials and allocation scope. These are single-thread CPU measurements,
including input conversion and output storage, with equal outputs verified.

The same interval-cache comparison uses CPU geometry through the Warp API:
120 repeated queries, 15 trials and one CPU thread reduced median per-query
latency from 0.726 to 0.560 ms. Cached and uncached outputs were identical.
[Interval-query data](benchmarks/interval-query.json) records all trials, residuals
and the command. This measures the CPU query path, separately from native
Warp batch throughput.

See [asset provenance](assets.md) for the R800 model. Analytical references:
[Shimizu et al., 2008](https://doi.org/10.1109/TRO.2008.2003266) and
[Faria et al., 2018](https://doi.org/10.1016/j.mechmachtheory.2017.10.025).
