"""Input validation agrees across devices and recovers after rejected batches."""

import numpy as np
import pytest
import torch

from srs_solver import SRSSolver, SRSSolverCfg

BACKENDS = [(False, "cpu"), (True, "cpu")] + (
    [(True, "cuda")] if torch.cuda.is_available() else []
)
Q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize(
    "kind,message",
    [
        ("nan_target", "target_xpos.*finite"),
        ("inf_target", "target_xpos.*finite"),
        ("nan_seed", "qpos_seed.*finite"),
        ("bottom", "rigid"),
        ("reflection", "rigid"),
        ("scale", "rigid"),
        ("determinant", "rigid"),
        ("mixed", "target_xpos.*finite"),
        ("rigid_and_seed", "rigid"),
    ],
)
def test_rejected_batches_report_priority_and_do_not_poison_next_call(
    use_warp, device, kind, message
):
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 2, device, use_warp=use_warp)
    target = solver.get_fk(Q).repeat(2, 1, 1)
    seed = torch.tensor(np.tile(Q + 0.1, (2, 1)), dtype=torch.float32, device=device)
    bad_target, bad_seed = target.clone(), seed.clone()
    if kind in ("nan_target", "mixed"):
        bad_target[1, 0, 3] = torch.nan
    if kind == "inf_target":
        bad_target[0, 0, 0] = torch.inf
    if kind in ("nan_seed", "mixed", "rigid_and_seed"):
        bad_seed[0, 2] = torch.nan
    if kind == "bottom":
        bad_target[0, 3, 1] = 1e-3
    if kind in ("reflection", "mixed", "rigid_and_seed"):
        bad_target[0, :3, 0] *= -1
    if kind == "scale":
        bad_target[1, :3, :3] *= 1.0001
    if kind == "determinant":
        # Gram errors remain below 1e-6, while det(R)-1 exceeds 1e-6.
        bad_target[1, :3, :3] = torch.eye(3, device=device) * (1 + 4e-7)
    with pytest.raises(ValueError, match=message):
        solver.get_ik(bad_target, bad_seed, search_mode="global")
    empty, _ = solver.get_ik(np.empty((0, 4, 4)))
    assert empty.shape == (0,)
    ok, joints = solver.get_ik(target, seed, search_mode="global")
    assert bool(ok.all())
    torch.testing.assert_close(solver.get_fk(joints[:, 0]), target, atol=2e-5, rtol=0)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_optional_seed_and_strided_rigid_inputs(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 2, device, use_warp=use_warp)
    target = solver.get_fk(Q).repeat(4, 1, 1)[::2]
    assert not target.is_contiguous()
    ok, joints = solver.get_ik(target, search_mode="global")
    assert bool(ok.all())
    torch.testing.assert_close(solver.get_fk(joints[:, 0]), target, atol=2e-5, rtol=0)
    # A sub-tolerance perturbation is accepted by the rigid-input contract.
    target = target.clone()
    target[0, :3, :3] = torch.eye(3, device=device) * (1 + 2e-7)
    solver.get_ik(target, search_mode="global")
