# Comparison with the reference papers

The Warp implementation accelerates a sampled analytical IK solver.
**It is not a complete reproduction** of Shimizu (2008) or Faria (2018).

The [shared method comparison](https://github.com/chase6305/7DofSRSKinematics/blob/improve/readme-viser-clarity/docs/paper-comparison.md)
maps the reference equations to the NumPy core, identifies missing feasible-interval
and avoidance policies, and includes a reproducible missed-solution example.
Both READMEs retain the full paper citations.

## What changes in the Warp version

| Area | Implementation |
| --- | --- |
| Native Warp search | Enumerates eight S/E/W branches and `cfg.num_samples` uniformly spaced arm angles. Geometry is shared, but no analytical feasible intervals are computed. |
| Selection | Chooses the sampled candidate nearest the seed using weighted joint distance. `return_all_solutions=True` returns sample slots, not the continuous solution manifold. |
| Continuous mode | Available in the NumPy CPU fallback. Native Warp supports global sampled search and rejects `search_mode="continuous"`. |
| Exact configurations | `solve_configuration` and `solve_configurations` use CPU geometry, including when the solver device is CUDA. |
| Singular poses | Coupled outer-joint solutions, finite tolerances and full-SVD null-space projection handle selected singular cases. There is no interval-based singularity exclusion or Faria K/α update. |

The narrow-box example in the shared comparison was checked on the NumPy
fallback, Warp CPU and Warp CUDA: exact configuration succeeds, while a
73-sample global search fails. A failed sample search therefore does not certify
unreachability. Batch acceleration does not remove that mathematical limitation.

See [solver/search and selection](../srs_solver.py),
[candidate kernels](../warp_kernel.py) and the
[viewer motion model](../srs_warp_viser/model.py).
Current round-trip, cross-backend and Tape-gradient tests establish numerical
consistency; the papers' interval algorithms and experimental protocols remain
outside the implemented scope.
