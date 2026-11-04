"""Numerical contracts shared by the CPU fallback and Warp kernels."""

import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from srs_solver import SRSSolver, SRSSolverCfg

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BACKENDS = [(False, "cpu")] + [(True, device) for device in DEVICES]
Q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])


def make_solver(use_warp, device, samples=32):
    return SRSSolver(SRSSolverCfg(num_samples=samples), 1, device, use_warp=use_warp)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("first,middle,last", [(0, 1, 2), (4, 5, 6)])
def test_singular_pairs_respect_narrow_limits(use_warp, device, first, middle, last):
    solver = make_solver(use_warp, device)
    q = Q.copy()
    q[2] = 0.0  # Reference shoulder plane: psi is on the uniform sample grid.
    q[[first, middle, last]] = [0.8, 0, 0.8]
    lower, upper = q - 1e-4, q + 1e-4
    lower[[first, last]], upper[[first, last]] = 0, 1
    assert solver.set_position_limits(lower, upper)
    seed = q.copy()
    seed[[first, last]] = 0
    target = solver.get_fk(q)
    ok, solution = solver.solve_configuration(
        target[0], solver.get_configuration(q), seed
    )
    assert ok
    torch.testing.assert_close(
        solution[[first, last]],
        torch.tensor([0.8, 0.8], device=device),
        atol=2e-6,
        rtol=0,
    )
    # Both singular examples lie on psi=-pi or psi=0, included by this grid.
    success, solutions = solver.get_ik(target, seed, search_mode="global")
    assert bool(success.all())
    torch.testing.assert_close(
        solver.get_fk(solutions[:, 0]), target, atol=5e-6, rtol=0
    )


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_periodic_limits_and_seed_fast_path(use_warp, device):
    solver = make_solver(use_warp, device)
    lower, upper = np.full(7, 2 * np.pi - 2), np.full(7, 2 * np.pi + 2)
    solver.set_position_limits(lower, upper)
    q = Q + 2 * np.pi
    target = solver.get_fk(q)
    for seed in (Q, q + 0.05):
        success, solutions = solver.get_ik(target, seed, search_mode="global")
        assert bool(success.all())
        result = solutions[:, 0]
        assert bool((result >= solver.lower_qpos_limits).all())
        assert bool((result <= solver.upper_qpos_limits).all())
        torch.testing.assert_close(solver.get_fk(result), target, atol=2e-5, rtol=0)
    solver.set_position_limits(np.zeros(7), np.zeros(7))
    success, _ = solver.get_ik(target, q)
    assert not bool(success.any())


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_analytical_jacobian_with_frames_and_reversed_joints(use_warp, device):
    cfg = SRSSolverCfg()
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    cfg.dh_params[:, 3] = np.linspace(-0.15, 0.15, 7)
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.3, -0.1, 0.2]
    cfg.T_e_oe[:3, 3] = [0.03, 0, 0.04]
    cfg.tcp[:3, 3] = [0.06, -0.02, 0.08]
    solver = SRSSolver(cfg, 3, device, use_warp=use_warp)
    joints = np.array([Q, -Q, np.zeros(7)])
    actual = solver.get_jacobian(joints).cpu().numpy()
    step = 1e-6
    # Use double-precision scalar FK, independent of the analytical Jacobian.
    for row, q in enumerate(joints):
        rotation = solver._geometry_impl._get_fk(q)[:3, :3]
        for index in range(7):
            delta = np.eye(7)[index] * step
            plus, minus = (
                solver._geometry_impl._get_fk(q + delta),
                solver._geometry_impl._get_fk(q - delta),
            )
            linear = (plus[:3, 3] - minus[:3, 3]) / (2 * step)
            omega = ((plus[:3, :3] - minus[:3, :3]) / (2 * step)) @ rotation.T
            expected = np.r_[linear, omega[2, 1], omega[0, 2], omega[1, 0]]
            np.testing.assert_allclose(
                actual[row, :, index], expected, atol=5e-7, rtol=0
            )
    direction = solver.get_null_space_direction(joints)
    assert (
        float(
            torch.linalg.vector_norm(solver.get_jacobian(joints) @ direction[..., None])
        )
        < 1e-6
    )


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_empty_strided_and_invalid_batches(use_warp, device):
    solver = make_solver(use_warp, device)
    assert solver.get_fk(np.empty((0, 7))).shape == (0, 4, 4)
    assert solver.get_jacobian(np.empty((0, 7))).shape == (0, 6, 7)
    assert solver.get_all_fk_mat(np.empty((0, 7))).shape == (0, 8, 4, 4)
    for all_solutions in (False, True):
        success, solutions = solver.get_ik(
            np.empty((0, 4, 4)), np.empty((0, 7)), return_all_solutions=all_solutions
        )
        assert success.shape == (0,)
        assert solutions.shape == (
            0,
            8 * solver.cfg.num_samples if all_solutions else 1,
            7,
        )
    target = solver.get_fk(Q).repeat(3, 1, 1)[::2]
    seed = torch.as_tensor(Q, dtype=torch.float32, device=device).repeat(4, 1)[::2]
    success, solution = solver.get_ik(target, seed + 0.02, search_mode="global")
    assert bool(success.all())
    preserved = solution.clone()
    solver.get_ik(target, seed - 0.02, search_mode="global")
    torch.testing.assert_close(solution, preserved)
    for malformed in (np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1])):
        with pytest.raises(ValueError):
            solver.get_ik(malformed, Q)
        with pytest.raises(ValueError):
            solver.set_tcp(malformed)
    with pytest.raises(ValueError):
        solver.get_jacobian(np.full(7, np.nan))
    for malformed in (np.zeros((2, 6)), np.zeros((1, 1, 7)), np.full(7, np.nan)):
        with pytest.raises(ValueError):
            solver.get_all_fk_mat(malformed)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_batched_link_poses_with_frames_offsets_and_strides(use_warp, device):
    cfg = SRSSolverCfg()
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    cfg.dh_params[:, 3] = np.linspace(-0.15, 0.15, 7)
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.3, -0.1, 0.2]
    cfg.T_e_oe[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cfg.T_e_oe[:3, 3] = [0.03, 0, 0.04]
    solver = SRSSolver(cfg, 1, device, use_warp=use_warp)
    tcp = np.eye(4)
    tcp[:3, 3] = [0.06, -0.02, 0.08]
    solver.set_tcp(tcp)
    joints = torch.tensor(np.array([Q, Q, -Q, -Q]), dtype=torch.float32, device=device)[
        ::2
    ]
    assert not joints.is_contiguous()
    actual = solver.get_all_fk_mat(joints)
    assert actual.shape == (2, 8, 4, 4)
    for row, q in enumerate(joints.cpu().numpy()):
        pose = cfg.T_b_ob.copy()
        expected = []
        for joint, (d, alpha, a, offset) in enumerate(cfg.dh_params):
            theta = q[joint] * cfg.rotation_directions[joint] + offset
            ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
            pose = pose @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
            expected.append(pose)
        expected.append(pose @ cfg.T_e_oe @ tcp)
        np.testing.assert_allclose(actual[row].cpu(), expected, atol=4e-7, rtol=0)
    torch.testing.assert_close(actual[:, -1], solver.get_fk(joints), atol=4e-7, rtol=0)
    preserved = actual.clone()
    solver.get_all_fk_mat(Q + 0.1)
    torch.testing.assert_close(actual, preserved)
    torch.testing.assert_close(
        solver.get_all_fk_mat(Q)[0], actual[0], atol=4e-7, rtol=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 1.5},
        {"num_samples": True},
        {"ik_nearest_weight": [-1] * 7},
        {"ik_nearest_weight": [np.nan] * 7},
        {"rotation_directions": [0] * 7},
        {"T_b_ob": np.diag([2, 1, 1, 1])},
    ],
)
def test_config_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SRSSolverCfg(**kwargs)


def test_explicit_zero_limits_are_not_replaced_by_defaults():
    cfg = SRSSolverCfg(qpos_limits=np.zeros((7, 2)))
    np.testing.assert_array_equal(cfg.qpos_limits, np.zeros((7, 2)))


@pytest.fixture
def reference_class():
    root = Path(
        os.environ.get(
            "SRS_NUMPY_REPO",
            str(Path(__file__).resolve().parents[2] / "7DofSRSKinematics"),
        )
    )
    if not (root / "src" / "kuka_iiwa_solver.py").is_file():
        pytest.skip("Set SRS_NUMPY_REPO to run the cross-repository comparisons")
    sys.path.insert(0, str(root / "src"))
    try:
        from kuka_iiwa_solver import KUKAiiwaSolver as Reference

        yield Reference
    finally:
        sys.path.remove(str(root / "src"))


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_matches_numpy_repository_across_all_branches(
    reference_class, use_warp, device
):
    ref = reference_class()
    cfg = SRSSolverCfg(num_samples=32)
    cfg.qpos_limits = np.column_stack(
        (ref.lower_position_limits, ref.upper_position_limits)
    )
    cfg.ik_nearest_weight = np.array([1, 2, 1, 3, 1, 2, 1])
    solver = SRSSolver(cfg, 8, device, use_warp=use_warp)
    ref.set_ik_nearst_weight(cfg.ik_nearest_weight)
    joints = np.tile(Q, (8, 1))
    joints[:, [1, 3, 5]] = np.array(list(product((-1, 1), repeat=3))) * np.abs(
        Q[[1, 3, 5]]
    )
    targets = ref.get_fk(joints)
    np.testing.assert_allclose(solver.get_fk(joints).cpu(), targets, atol=3e-7, rtol=0)
    np.testing.assert_allclose(
        solver.get_all_fk_mat(joints).cpu(),
        ref.get_all_fk_mat(joints),
        atol=4e-7,
        rtol=0,
    )
    for q, target in zip(joints, targets):
        np.testing.assert_allclose(
            solver.get_jacobian(q).cpu(), ref.get_jacobian(q), atol=5e-7, rtol=0
        )
        rc, wc = ref.get_configuration(q), solver.get_configuration(q)
        assert (rc.shoulder, rc.elbow, rc.wrist) == (wc.shoulder, wc.elbow, wc.wrist)
        assert rc.redundancy == pytest.approx(wc.redundancy, abs=1e-12)
        from srs_analytical_solver import SRSConfiguration as ReferenceConfiguration

        from srs_solver import SRSConfiguration

        requests = [
            (branch, rc.redundancy + delta)
            for branch in product((-1, 1), repeat=3)
            for delta in (-0.2, 0, 0.2)
        ]
        ref_ok, ref_q = ref.solve_configurations(
            target,
            [ReferenceConfiguration(*branch, psi) for branch, psi in requests],
            q,
        )
        warp_ok, warp_q = solver.solve_configurations(
            target, [SRSConfiguration(*branch, psi) for branch, psi in requests], q
        )
        np.testing.assert_array_equal(warp_ok.cpu(), ref_ok)
        np.testing.assert_allclose(warp_q.cpu(), ref_q, atol=2e-5, rtol=0)
    seeds = joints + 0.03
    success, solutions = solver.get_ik(targets, seeds, search_mode="global")
    assert bool(success.all())
    for target, seed, actual in zip(targets, seeds, solutions[:, 0].cpu().numpy()):
        ok, expected = ref.get_ik(target, seed, num_samples=32, search_mode="global")
        assert ok
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
def test_reversed_joint_configuration_and_weighted_singular_solution(use_warp, device):
    cfg = SRSSolverCfg(num_samples=32)
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    cfg.ik_nearest_weight = np.array([1, 1, 3, 1, 1, 1, 1])
    solver = SRSSolver(cfg, 1, device, use_warp=use_warp)
    q = np.array([-0.8, 0, -0.8, 1, 0.4, 0.8, 0.2])
    lower, upper = q - 1e-4, q + 1e-4
    lower[[0, 2]], upper[[0, 2]] = -1, 0
    solver.set_position_limits(lower, upper)
    seed = q.copy()
    seed[[0, 2]] = 0
    target = solver.get_fk(q)
    configuration = solver.get_configuration(q)
    assert configuration.shoulder == 1
    ok, result = solver.solve_configuration(target[0], configuration, seed)
    assert ok
    np.testing.assert_allclose(result[[0, 2]].cpu(), [-1, -0.6], atol=3e-6)
    success, result = solver.get_ik(target, seed, search_mode="global")
    assert bool(success.all())
    np.testing.assert_allclose(result[0, 0, [0, 2]].cpu(), [-1, -0.6], atol=3e-6)


def test_bundled_assets_and_demo_model_paths():
    from xml.etree import ElementTree

    from example.demo import build_solver_cfg
    from srs_solver import get_iiwa_urdf_path

    path = Path(get_iiwa_urdf_path())
    assert path.is_file()
    for mesh in ElementTree.parse(path).findall(".//mesh"):
        assert (path.parent / mesh.get("filename")).is_file()
    assert build_solver_cfg().urdf_path == str(path)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("count", [0, 1, 17])
def test_numpy_batch_tcp_matches_warp_with_frames_and_strides(
    reference_class, use_warp, device, count
):
    ref = reference_class()
    cfg = SRSSolverCfg()
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.3, -0.2, 0.1]
    cfg.T_e_oe[:3, 3] = [0.02, 0.01, -0.01]
    cfg.tcp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cfg.tcp[:3, 3] = [-0.01, 0.03, 0.02]
    cfg.dh_params[:, 3] = np.linspace(-0.1, 0.1, 7)
    ref.world_to_base = cfg.T_b_ob.copy()
    ref.set_tcp(cfg.T_e_oe @ cfg.tcp)
    ref.dh_params[:, 3] = cfg.dh_params[:, 3]
    solver = SRSSolver(cfg, max(1, count), device, use_warp=use_warp)
    q = np.random.default_rng(459).uniform(-1, 1, (count * 2, 7))[::2]
    reference, actual = ref.get_fk(q), solver.get_fk(q).cpu().numpy()
    assert reference.shape == actual.shape == (count, 4, 4)
    np.testing.assert_allclose(actual, reference, atol=4e-7, rtol=0)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("return_all", [False, True])
def test_nearest_weight_ranking_ignores_finite_common_scale(
    use_warp, device, scale, return_all
):
    weights = np.array([1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0])
    outputs = []
    for multiplier in (1.0, scale):
        solver = SRSSolver(
            SRSSolverCfg(num_samples=32, ik_nearest_weight=weights * multiplier),
            1,
            device,
            use_warp=use_warp,
        )
        target = solver.get_fk(Q)
        with np.errstate(over="raise", invalid="raise"):
            ok, solutions = solver.get_ik(
                target, Q + 0.2, return_all_solutions=return_all, search_mode="global"
            )
        assert bool(ok.all())
        outputs.append(solutions)
        np.testing.assert_array_equal(
            solver.cfg.ik_nearest_weight, weights * multiplier
        )
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-6, rtol=0)
