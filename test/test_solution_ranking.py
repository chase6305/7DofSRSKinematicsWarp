"""Public candidate ordering and tie behavior for the fused score kernel."""

import numpy as np
import pytest
import torch

from srs_solver import SRSSolver, SRSSolverCfg

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weights", [[1] * 7, [1, 2, 0, 3, 1, 0, 4], [0] * 7])
def test_weighted_nearest_all_order_and_failed_rows(device, weights):
    cfg = SRSSolverCfg(num_samples=32, ik_nearest_weight=weights, sort_ik=False)
    solver = SRSSolver(cfg, 3, device, use_warp=True)
    q = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]], device=device).repeat(
        3, 1
    )
    q[1] *= -1
    targets = solver.get_fk(q)
    targets[2, 0, 3] = 10
    seed = q + 0.23
    ok, candidates = solver.get_ik(targets, seed, return_all_solutions=True)
    assert ok.tolist() == [True, True, False]
    preserved = candidates.clone()
    # These nonzero targets have no valid all-zero joint solution.
    valid = candidates.abs().sum(dim=2) > 0
    cost = torch.linalg.vector_norm(
        (candidates - seed[:, None]) * torch.tensor(weights, device=device), dim=2
    ).masked_fill(~valid, torch.inf)
    order = cost.argsort(dim=1, stable=True)
    expected = candidates.gather(1, order[:, :, None].expand(-1, -1, 7))

    cfg.sort_ik = True
    sorted_ok, sorted_candidates = solver.get_ik(
        targets, seed, return_all_solutions=True
    )
    nearest_ok, nearest = solver.get_ik(targets, seed)
    torch.testing.assert_close(sorted_ok, ok)
    torch.testing.assert_close(nearest_ok, ok)
    torch.testing.assert_close(sorted_candidates, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(nearest, expected[:, :1], atol=1e-6, rtol=0)
    assert not bool(sorted_candidates[2].any())
    assert not bool(nearest[2].any())
    torch.testing.assert_close(candidates, preserved)

    # Replacing the cached workspace with another batch must not alter outputs.
    solver.get_ik(targets[:1], seed[:1])
    torch.testing.assert_close(sorted_candidates, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(nearest, expected[:, :1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_weight_edits_refresh_cached_scores(device):
    cfg = SRSSolverCfg(num_samples=32, sort_ik=False)
    solver = SRSSolver(cfg, 1, device, use_warp=True)
    q = torch.tensor([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], device=device)
    target, seed = solver.get_fk(q), q + 0.23
    _, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    valid = candidates.abs().sum(dim=2) > 0
    cfg.sort_ik = True
    solver.get_ik(target, seed)  # Populate the reusable score buffer.
    weights = np.array([100.0, 1, 0, 1, 1, 0, 1])
    cfg.ik_nearest_weight[:] = weights * 1e-200
    cost = torch.linalg.vector_norm(
        (candidates - seed[None, None])
        * torch.tensor(weights, dtype=torch.float32, device=device),
        dim=2,
    ).masked_fill(~valid, torch.inf)
    expected = candidates[:, cost.argmin(dim=1).item() :][:, :1]
    ok, nearest = solver.get_ik(target, seed)
    assert bool(ok.all())
    torch.testing.assert_close(nearest, expected, atol=1e-6, rtol=0)
