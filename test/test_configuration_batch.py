"""Exact configuration batches use the same scalar geometry on each device."""

from itertools import product

import numpy as np
import pytest
import torch

from srs_solver import SRSConfiguration, SRSSolver, SRSSolverCfg

BACKENDS = [(False, "cpu"), (True, "cpu")] + (
    [(True, "cuda")] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("middle", [None, 1, 5])
def test_configuration_batch_matches_scalar_with_reversed_joints_and_frames(
    use_warp, device, middle, monkeypatch
):
    cfg = SRSSolverCfg()
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    cfg.ik_nearest_weight = np.array([1, 2, 3, 1, 3, 2, 1])
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.2, -0.1, 0.3]
    cfg.tcp[:3, 3] = [0.03, -0.02, 0.05]
    solver = SRSSolver(cfg, 1, device, use_warp=use_warp)
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    if middle is not None:
        q[middle] = 0
    target = solver._geometry_impl._get_fk(q)
    psi = solver.get_arm_angle(q)
    configs = [
        SRSConfiguration(*branch, psi + delta)
        for delta in (0.2, 0, -0.2, 0)
        for branch in product((-1, 1), repeat=3)
    ]
    expected = [solver.solve_configuration(target, c, q) for c in configs]
    calls = []
    prepare = solver._geometry_impl._prepare_ik

    def counted(local, elbow):
        calls.append(elbow)
        return prepare(local, elbow)

    monkeypatch.setattr(solver._geometry_impl, "_prepare_ik", counted)
    ok, joints = solver.solve_configurations(target, iter(configs), q)
    assert sorted(calls) == [-1, 1]
    assert joints.device.type == device
    assert ok.dtype == torch.bool
    np.testing.assert_array_equal(ok.cpu(), [item[0] for item in expected])
    assert bool(ok.any())
    for valid, actual, (_, ref) in zip(ok, joints, expected):
        if bool(valid):
            torch.testing.assert_close(actual, ref, atol=1e-6, rtol=0)
            fk = solver.get_fk(actual)[0].cpu().numpy()
            assert np.linalg.norm(fk[:3, 3] - target[:3, 3]) < 2e-6
            assert np.linalg.norm(fk[:3, :3] - target[:3, :3]) < 3e-6
        else:
            assert not bool(actual.any())
    preserved = joints.clone()
    solver.solve_configurations(target, configs[:1], q + 0.02)
    torch.testing.assert_close(joints, preserved)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_empty_invalid_and_unreachable_batches(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(), 1, device, use_warp=use_warp)
    target = solver.get_fk(np.zeros(7))[0]
    ok, joints = solver.solve_configurations(target, [])
    assert ok.shape == (0,) and joints.shape == (0, 7)
    target[0, 3] = 10
    config = SRSConfiguration(1, 1, 1, 0)
    ok, joints = solver.solve_configurations(target, [config])
    assert not bool(ok.any()) and not bool(joints.any())
    with pytest.raises(TypeError):
        solver.solve_configurations(target, [config, None])
    with pytest.raises(ValueError):
        solver.solve_configurations(target, [], np.full(7, np.nan))
    with pytest.raises(ValueError):
        solver.solve_configurations(np.diag([2, 1, 1, 1]), [])
