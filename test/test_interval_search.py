"""Boundary-event search uses float64 geometry and returns device tensors."""

import numpy as np
import pytest
import torch
import warp as wp

from srs_solver import SRSConfiguration, SRSSolver, SRSSolverCfg

BACKENDS = [(False, "cpu"), (True, "cpu")] + (
    [(True, "cuda")] if torch.cuda.is_available() else []
)
Q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])


def assert_solution(solver, targets, seeds=None, mode="interval"):
    with np.errstate(all="raise"):
        ok, joints = solver.get_ik(targets, seeds, search_mode=mode)
    assert bool(ok.all())
    assert ok.device == joints.device == solver.device
    assert ok.dtype == torch.bool and joints.dtype == torch.float32
    assert joints.shape == (len(ok), 1, 7)
    assert bool((joints[:, 0] >= solver.lower_qpos_limits).all())
    assert bool((joints[:, 0] <= solver.upper_qpos_limits).all())
    actual = solver.get_fk(joints[:, 0])
    expected = torch.as_tensor(targets, dtype=torch.float32, device=solver.device)
    torch.testing.assert_close(actual, expected.reshape(-1, 4, 4), atol=2e-6, rtol=0)
    return joints


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_narrow_limits_known_target_and_optional_seed(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, device, use_warp=use_warp)
    solver.set_position_limits(Q - 1e-5, Q + 1e-5)
    target = solver.get_fk(Q)
    first = assert_solution(solver, target)
    torch.testing.assert_close(assert_solution(solver, target, np.zeros(7)), first)
    if not use_warp:
        assert_solution(solver, target, mode="continuous")


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_reversed_joints_offsets_frames_and_periodic_limits(use_warp, device):
    cfg = SRSSolverCfg(num_samples=2)
    cfg.rotation_directions = np.array([-1, 1, -1, -1, -1, 1, -1])
    cfg.dh_params[:, 3] = offsets = np.linspace(-0.2, 0.2, 7)
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.2, -0.1, 0.3]
    cfg.tcp[:3, 3] = [0.03, -0.02, 0.05]
    solver = SRSSolver(cfg, 1, device, use_warp=use_warp)
    q = (Q - offsets) * cfg.rotation_directions + 2 * np.pi
    solver.set_position_limits(q - 1e-4, q + 1e-4)
    target = solver.get_fk(q)
    assert_solution(solver, target, np.zeros(7))


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("middle", [None, 1, 5, "both", "straight"])
def test_singular_events_with_tight_limits(use_warp, device, middle):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, device, use_warp=use_warp)
    q = Q.copy()
    if middle in (1, 5):
        q[middle] = 0
    elif middle == "both":
        q[[1, 5]] = 0
    elif middle == "straight":
        q[[1, 3, 5]] = 0
    solver.set_position_limits(q - 2e-5, q + 2e-5)
    assert_solution(solver, solver.get_fk(q), np.ones(7))


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_strided_mixed_batches_empty_output_and_candidate_contract(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 2, device, use_warp=use_warp)
    solver.set_position_limits(Q - 1e-4, Q + 1e-4)
    target = solver.get_fk(Q).repeat(4, 1, 1)[::2]
    assert not target.is_contiguous()
    target[1, 0, 3] = 10
    ok, joints = solver.get_ik(target, search_mode="interval")
    assert ok.tolist() == [True, False]
    assert not bool(joints[1].any())
    preserved = joints.clone()
    assert_solution(solver, target[:1])
    torch.testing.assert_close(joints, preserved)
    ok, joints = solver.get_ik(np.empty((0, 4, 4)), search_mode="interval")
    assert ok.shape == (0,) and joints.shape == (0, 1, 7)
    assert joints.device == solver.device
    with pytest.raises(ValueError, match="one solution"):
        solver.get_ik(target, return_all_solutions=True, search_mode="interval")
    invalid = target[:1].clone()
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        solver.get_ik(invalid, search_mode="interval")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_interval_query_obeys_the_callers_cuda_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, "cuda", use_warp=True)
        solver.set_position_limits(Q - 1e-4, Q + 1e-4)
        target = solver.get_fk(torch.tensor(Q, dtype=torch.float32, device="cuda"))
        ok, joints = solver.get_ik(target, search_mode="interval")
        actual = solver.get_fk(joints[:, 0]).clone()
    assert bool(ok.all())
    torch.testing.assert_close(actual, target, atol=2e-6, rtol=0)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_repeated_target_respects_changed_limits_and_tcp(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, device, use_warp=use_warp)
    target = solver._geometry_impl._get_fk(Q)
    config = solver.get_configuration(Q)
    shifted = SRSConfiguration(
        config.shoulder, config.elbow, config.wrist, config.redundancy + 0.2
    )
    ok, other = solver.solve_configuration(target, shifted, Q)
    assert ok
    other = other.cpu().numpy()
    for q in (Q, other, Q):
        solver.set_position_limits(q - 1e-5, q + 2e-5)
        assert_solution(solver, target, np.zeros(7))
    tcp = np.eye(4)
    tcp[:3, 3] = [0.01, -0.02, 0.03]
    solver.set_tcp(tcp)
    assert_solution(solver, solver.get_fk(Q), np.zeros(7))


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_interval_queries_do_not_launch_native_kernels(device, monkeypatch):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, device, use_warp=True)
    solver.set_position_limits(Q - 1e-4, Q + 1e-4)
    target = solver._geometry_impl._get_fk(Q)

    def reject_native_launch(*args, **kwargs):
        raise AssertionError("interval queries use CPU geometry and validation")

    monkeypatch.setattr(wp, "launch", reject_native_launch)
    ok, joints = solver.get_ik(target, search_mode="interval")
    assert bool(ok.all()) and joints.device == solver.device


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("angle", [0, np.pi])
@pytest.mark.parametrize("width", [0, 1e-5])
def test_degenerate_elbow_with_a_narrow_joint_box(use_warp, device, angle, width):
    solver = SRSSolver(SRSSolverCfg(num_samples=2), 1, device, use_warp=use_warp)
    q = Q.copy()
    q[3] = angle
    solver.set_position_limits(q - width, q + width)
    target = solver._geometry_impl._get_fk(q)
    assert_solution(solver, target, np.zeros(7))
