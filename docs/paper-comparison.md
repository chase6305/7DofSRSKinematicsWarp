# Comparison with the reference papers

The Warp implementation accelerates a sampled analytical IK solver.
**It is not a complete reproduction** of Shimizu (2008) or Faria (2018).

The [shared method
comparison](https://github.com/chase6305/7DofSRSKinematics/blob/ff8456b83d4084391a0bfc7ab2e2acca8512b626/docs/paper-comparison.md)
maps the reference equations to the NumPy core, describes joint-boundary search, remaining
feasible-set
and avoidance gaps, and includes a reproducible missed-solution example.
Both READMEs retain the full paper citations.

## What changes in the Warp version

| Area | Implementation |
| --- | --- |
| Native Warp search | Enumerates eight S/E/W branches and `cfg.num_samples` uniformly spaced arm angles. Geometry is shared, but no analytical feasible intervals are computed. |
| Selection | Chooses the sampled candidate nearest the seed using weighted joint distance. `return_all_solutions=True` returns sample slots, not the continuous solution manifold. |
| Boundary search | `search_mode="interval"` checks analytical limit events and cell midpoints through float64 CPU geometry, then returns device tensors. It recovers narrow ranges without increasing the uniform grid. |
| Continuous mode | Available in the NumPy CPU fallback, with boundary search as the final fallback. Native Warp rejects `search_mode="continuous"`. |
| Arithmetic | Native kernels and public tensor results use float32. Exact and interval geometry uses NumPy float64; FK is checked on the float32 joints that will be returned. |
| Exact configurations | `solve_configuration` and `solve_configurations` use CPU geometry, including when the solver device is CUDA. |
| Singular poses | IK uses coupled outer-joint solutions; differential motion uses full-SVD null-space projection. There is no interval-based singularity exclusion or Faria K/α update. |

The narrow-box example in the shared comparison was checked on the NumPy
fallback, Warp CPU and Warp CUDA: exact configuration succeeds, while a
73-sample global search fails. Explicit interval search now succeeds on each backend. A failed
sample search therefore does not certify
unreachability. Batch acceleration does not remove that mathematical limitation.

See [solver/search and selection](../srs_solver.py),
[candidate kernels](../warp_kernel.py) and the
[viewer motion model](../srs_warp_viser/model.py).
Current round-trip, cross-backend and Tape-gradient tests establish numerical
consistency; the papers' full feasible-set/avoidance policies and experimental protocols
remain outside the implemented scope.
