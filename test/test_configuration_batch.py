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
@pytest.mark.parametrize("query", ["scalar", "batch", "interval"])
def test_fk_validation_uses_returned_float32_joints(use_warp, device, query):
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 1, device, use_warp=use_warp)
    nominal = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])

    def solve(q):
        target = solver._geometry_impl._get_fk(q)
        config = solver.get_configuration(q)
        if query == "scalar":
            ok, joints = solver.solve_configuration(target, config, q)
            if not ok:
                assert joints is None
                return False
        elif query == "batch":
            ok, joints = solver.solve_configurations(target, [config], q)
            if not bool(ok[0]):
                assert not bool(joints.any())
                return False
            joints = joints[0]
        else:
            ok, joints = solver.get_ik(target, q, search_mode="interval")
            if not bool(ok[0]):
                assert not bool(joints.any())
                return False
            joints = joints[0, 0]
        actual = solver._geometry_impl._get_fk(joints.cpu().numpy())
        np.testing.assert_allclose(actual, target, atol=1e-4, rtol=0)
        return True

    shifted = nominal + 2 * np.pi * 100000
    solver.set_position_limits(shifted - 0.1, shifted + 0.1)
    # A float64 solution can lose centimetres of accuracy when stored as
    # float32 at this magnitude. Success must describe the returned tensor.
    solve(shifted)
    solver.set_position_limits(nominal - 0.1, nominal + 0.1)
    assert solve(nominal)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize(
    "middle,value", [(None, 0), (1, 0), (5, 0), (1, np.pi), (5, np.pi)]
)
def test_configuration_batch_matches_scalar_with_reversed_joints_and_frames(
    use_warp, device, middle, value, monkeypatch
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
        q[middle] = value
    if value:
        solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    target = solver._geometry_impl._get_fk(q)
    psi = solver.get_arm_angle(q)
    configs = [
        SRSConfiguration(*branch, psi + delta)
        for delta in (0.2, 0, -0.2, 0)
        for branch in product((-1, 1), repeat=3)
    ]
    configs = [configs[i] for i in np.random.default_rng(7).permutation(len(configs))]
    expected = [solver.solve_configuration(target, c, q) for c in configs]
    calls = []
    prepare = solver._geometry_impl._prepare_ik
    decompose = solver._geometry_impl._candidate_joint_angles
    decompositions = []

    def counted(local, elbow):
        calls.append(elbow)
        return prepare(local, elbow)

    monkeypatch.setattr(solver._geometry_impl, "_prepare_ik", counted)

    def counted_decomposition(prepared, angles, branches, seed):
        decompositions.append(np.asarray(branches).copy())
        return decompose(prepared, angles, branches, seed)

    monkeypatch.setattr(
        solver._geometry_impl, "_candidate_joint_angles", counted_decomposition
    )
    ok, joints = solver.solve_configurations(target, iter(configs), q)
    assert sorted(calls) == [-1, 1]
    assert len(decompositions) == 2
    assert all(len(np.unique(branches & 2)) == 1 for branches in decompositions)
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


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_exact_arm_angle_precision_is_independent_of_scalar_storage(
    use_warp, device, dtype
):
    solver = SRSSolver(SRSSolverCfg(), 1, device, use_warp=use_warp)
    solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver._geometry_impl._get_fk(q)
    branch = solver.get_configuration(q)
    for phase in (dtype(0.1), dtype(-2.7), dtype(65504)):
        requested = SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, phase)
        canonical = SRSConfiguration(
            branch.shoulder, branch.elbow, branch.wrist, float(phase)
        )
        expected_ok, expected = solver.solve_configuration(target, canonical, q)
        ok, actual = solver.solve_configuration(target, requested, q)
        assert expected_ok and ok
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        mask, joints = solver.solve_configurations(target, [requested], q)
        assert bool(mask[0])
        torch.testing.assert_close(joints[0], expected, atol=0, rtol=0)
        np.testing.assert_allclose(
            solver._geometry_impl._get_fk(actual.cpu().numpy()),
            target,
            atol=2e-6,
            rtol=0,
        )


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_request_cache_detects_replaced_records_and_invalid_list_entries(
    use_warp, device
):
    solver = SRSSolver(SRSSolverCfg(), 1, device, use_warp=use_warp)
    solver.set_position_limits(np.full(7, -4.0), np.full(7, 4.0))
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver._geometry_impl._get_fk(q)
    branch = solver.get_configuration(q)
    requests = [
        SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, np.float16(0.1))
    ]
    ok, first = solver.solve_configurations(target, requests, q)
    assert bool(ok[0])
    requests[0] = SRSConfiguration(branch.shoulder, branch.elbow, branch.wrist, 0.1)
    ok, second = solver.solve_configurations(target, requests, q)
    assert bool(ok[0]) and float(torch.linalg.vector_norm(first - second)) > 1e-6
    expected_ok, expected = solver.solve_configuration(target, requests[0], q)
    assert expected_ok
    torch.testing.assert_close(second[0], expected, atol=0, rtol=0)
    requests[0] = np.zeros(7)
    with pytest.raises(TypeError, match="SRSConfiguration"):
        solver.solve_configurations(target, requests, q)
