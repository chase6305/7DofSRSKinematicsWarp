"""Large finite seeds must never make invalid candidates win the distance sort."""

import numpy as np
import pytest
import torch

from srs_solver import KUKAiiwaSolver, SRSSolverCfg

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("magnitude", [1e20, 3e38])
@pytest.mark.parametrize("weighted", [False, True])
def test_large_seeds_keep_valid_candidates_first(device, magnitude, weighted):
    cfg = SRSSolverCfg(num_samples=17, sort_ik=False)
    limits = np.array([170, 120, 170, 120, 170, 120, 175])
    cfg.qpos_limits = np.deg2rad(np.column_stack((-limits, limits)))
    if weighted:
        cfg.ik_nearest_weight = np.array([1, 0, 2, 0, 3, 0, 4])
    solver = KUKAiiwaSolver(device=device, cfg=cfg, use_warp=True)
    q = torch.tensor([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], device=device)
    target = solver.get_fk(q)
    seed = torch.full_like(q, magnitude)
    ok, original = solver.get_ik(target, seed, return_all_solutions=True)
    valid = original.abs().sum(dim=2) > 0
    assert bool(ok.all()) and not bool(valid[0, 0])
    first = original[0, torch.nonzero(valid[0])[0, 0]].clone()

    cfg.sort_ik = True
    ok, nearest = solver.get_ik(target, seed, search_mode="global")
    assert bool(ok.all())
    # At these seed magnitudes float32/float64 cannot distinguish O(1) joint
    # changes in the distances. The first valid candidate wins stable ties.
    torch.testing.assert_close(nearest[0, 0], first, atol=0, rtol=0)
    torch.testing.assert_close(solver.get_fk(nearest[:, 0]), target, atol=2e-5, rtol=0)
    ok, ordered = solver.get_ik(target, seed, return_all_solutions=True)
    count = int(valid.sum())
    assert bool(ok.all()) and bool((ordered[:, :count].abs().sum(dim=2) > 0).all())
    assert not bool(ordered[:, count:].any())
    torch.testing.assert_close(ordered[0, 0], first, atol=0, rtol=0)
