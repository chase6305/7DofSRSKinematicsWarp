import numpy as np
import pytest
import torch

from srs_solver import KUKAiiwaSolver, SRSConfiguration, SRSSolver, SRSSolverCfg


@pytest.mark.parametrize(
    "use_warp,device",
    [(False, "cpu"), (True, "cpu")]
    + ([(True, "cuda")] if torch.cuda.is_available() else []),
)
@pytest.mark.parametrize("autocast", [False, True])
def test_ik_is_independent_of_torch_default_dtype(use_warp, device, autocast):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        solver = SRSSolver(SRSSolverCfg(num_samples=32), 1, device, use_warp)
        for return_all in (False, True):
            success, empty = solver.get_ik(
                np.empty((0, 4, 4)), return_all_solutions=return_all
            )
            assert success.dtype == torch.bool and success.shape == (0,)
            slots = 8 * solver.cfg.num_samples if return_all else 1
            assert empty.dtype == torch.float32 and empty.shape == (0, slots, 7)
            assert empty.device == solver.device

        joints = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]])
        target = solver.get_fk(joints)
        with torch.autocast(device_type=device, enabled=autocast):
            success, solution = solver.get_ik(
                target.double(), joints + 0.02, search_mode="global"
            )
        assert bool(success.all()) and solution.dtype == torch.float32
        torch.testing.assert_close(
            solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=0
        )
    finally:
        torch.set_default_dtype(previous)


def test_batched_fk_matches_scalar_reference():
    solver = SRSSolver(SRSSolverCfg(), num_envs=3, device="cpu")
    joints = np.array(
        [
            [0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2],
            [-0.8, 0.4, -0.3, -1.0, 0.5, -0.7, 0.6],
            [0.2, -0.4, 0.3, 0.9, -0.2, 0.6, -0.1],
        ],
        dtype=np.float32,
    )
    actual = solver.get_fk(joints).numpy()
    expected = np.stack([solver.impl._get_fk(row) for row in joints])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_batched_fk_validation():
    solver = SRSSolver(SRSSolverCfg(), num_envs=1, device="cpu")
    with pytest.raises(ValueError):
        solver.get_fk(torch.zeros(6))
    with pytest.raises(ValueError):
        solver.get_fk(torch.full((1, 7), torch.nan))
    with pytest.raises(ValueError):
        solver.get_ik(torch.eye(3))
    with pytest.raises(ValueError):
        solver.get_ik(torch.eye(4).repeat(2, 1, 1), torch.zeros(1, 7))


def test_config_validation():
    with pytest.raises(ValueError):
        SRSSolverCfg(dh_params=np.zeros((6, 4)), link_lengths=np.ones(4))
    with pytest.raises(ValueError):
        SRSSolverCfg(num_samples=0)
    mutated = SRSSolverCfg()
    mutated.qpos_limits = np.zeros((6, 2))
    with pytest.raises(ValueError):
        SRSSolver(mutated, num_envs=1, device="cpu")
    with pytest.raises(ValueError):
        SRSSolver(SRSSolverCfg(), num_envs=0, device="cpu")


def test_limits_jacobian_and_null_space_api():
    solver = SRSSolver(SRSSolverCfg(), num_envs=1, device="cpu")
    assert solver.get_dof() == 7
    assert solver.set_position_limits([-2.0] * 7, [2.0] * 7)
    assert not solver.set_position_limits([-2.0] * 6, [2.0] * 6)
    limits = solver.get_position_limits()
    np.testing.assert_array_equal(limits["lower"], np.full(7, -2.0))

    joints = torch.tensor([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    jacobian = solver.get_jacobian(joints)
    assert jacobian.shape == (6, 7)
    direction = solver.get_null_space_direction(joints)
    assert torch.linalg.vector_norm(jacobian @ direction) < 1e-3
    velocity = solver.get_null_space_velocity(joints, torch.linspace(-1.0, 1.0, 7))
    assert velocity.shape == (7,)
    assert torch.linalg.vector_norm(jacobian @ velocity) < 1e-3


def test_stationary_seed_fast_path_and_failure_shapes():
    solver = SRSSolver(SRSSolverCfg(num_samples=9), num_envs=1, device="cpu")
    zero = torch.zeros(1, 7)
    success, solution = solver.get_ik(solver.get_fk(zero), zero)
    assert success.tolist() == [True]
    assert solution.shape == (1, 1, 7)
    torch.testing.assert_close(solution[:, 0], zero)

    unreachable = torch.eye(4).unsqueeze(0)
    unreachable[:, 0, 3] = 10.0
    success, solution = solver.get_ik(unreachable, zero)
    assert success.tolist() == [False]
    assert solution.shape == (1, 1, 7)
    success, solutions = solver.get_ik(unreachable, zero, return_all_solutions=True)
    assert success.tolist() == [False]
    assert solutions.shape == (1, 8 * solver.cfg.num_samples, 7)


@pytest.mark.parametrize(
    "use_warp,device",
    [(False, "cpu"), (True, "cpu")]
    + ([(True, "cuda")] if torch.cuda.is_available() else []),
)
def test_quantized_fully_extended_fk_target_can_be_solved(use_warp, device):
    solver = SRSSolver(SRSSolverCfg(num_samples=73), 1, device, use_warp)
    q = torch.zeros((1, 7), device=device)
    target = solver.get_fk(q)
    success, solution = solver.get_ik(target, q + 0.02, search_mode="global")
    assert bool(success.all())
    actual = solver.get_fk(solution[:, 0])
    torch.testing.assert_close(actual, target, atol=2e-6, rtol=0)


def test_kuka_convenience_solver():
    solver = KUKAiiwaSolver(device="cpu")
    limits = solver.get_position_limits()
    np.testing.assert_allclose(
        np.rad2deg(limits["upper"]), [170, 120, 170, 120, 170, 120, 175]
    )
    assert solver.urdf_path.endswith("urdf/lbr_iiwa7_r800.urdf")


def test_explicit_configuration_round_trip():
    solver = KUKAiiwaSolver(device="cpu")
    joints = torch.tensor([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(joints)[0]
    configuration = solver.get_configuration(joints)
    assert isinstance(configuration, SRSConfiguration)
    success, solution = solver.solve_configuration(target, configuration)
    assert success
    torch.testing.assert_close(solver.get_fk(solution)[0], target, atol=1e-4, rtol=0.0)

    with pytest.raises(ValueError):
        SRSConfiguration(0, 1, 1, 0.0)


def test_tcp_round_trip():
    solver = KUKAiiwaSolver(device="cpu")
    tcp = np.eye(4)
    tcp[:3, 3] = [0.02, -0.01, 0.08]
    solver.set_tcp(tcp)
    np.testing.assert_array_equal(solver.get_tcp(), tcp)
    joints = torch.tensor([[0.2, -0.4, 0.3, 0.9, -0.2, 0.6, -0.1]])
    target = solver.get_fk(joints)
    success, solution = solver.get_ik(target, joints + 0.01)
    assert success.tolist() == [True]
    torch.testing.assert_close(
        solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=0.0
    )


def test_cpu_continuous_seed_path_avoids_global_grid(monkeypatch):
    solver = KUKAiiwaSolver(device="cpu")
    joints = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]])
    target = solver.get_fk(joints)
    seed = joints + 0.01
    calls = 0
    original = solver.impl._get_each_ik

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver.impl, "_get_each_ik", counted)
    success, solution = solver.get_ik(target, seed)
    assert success.tolist() == [True]
    assert calls == 1
    torch.testing.assert_close(
        solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=0.0
    )


def test_return_all_solutions_runs_global_search():
    solver = KUKAiiwaSolver(device="cpu", cfg=SRSSolverCfg(num_samples=17))
    joints = torch.tensor([[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]])
    success, solutions = solver.get_ik(
        solver.get_fk(joints), joints, return_all_solutions=True
    )
    assert success.tolist() == [True]
    assert solutions.shape == (1, 8 * solver.cfg.num_samples, 7)
    assert int((solutions.abs().sum(dim=2) > 0.0).sum()) > 1


def test_warp_cpu_batched_fk_ik_round_trip():
    solver = SRSSolver(
        SRSSolverCfg(num_samples=25), num_envs=4, device="cpu", use_warp=True
    )
    joints = torch.tensor(
        [
            [0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2],
            [-0.8, 0.4, -0.3, -1.0, 0.5, -0.7, 0.6],
            [0.2, -0.4, 0.3, 0.9, -0.2, 0.6, -0.1],
            [-0.4, 0.6, -0.2, -0.9, 0.5, -0.7, 0.3],
        ]
    )
    targets = solver.get_fk(joints)
    success, solutions = solver.get_ik(targets, joints + 0.01)
    assert bool(success.all())
    torch.testing.assert_close(
        solver.get_fk(solutions[:, 0]), targets, atol=2e-4, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_batched_fk_ik_round_trip():
    cfg = SRSSolverCfg(num_samples=73)
    solver = SRSSolver(cfg, num_envs=16, device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(19)
    joints = (torch.rand((16, 7), generator=generator, device="cuda") - 0.5) * 2.0
    targets = solver.get_fk(joints)
    success, solutions = solver.get_ik(targets, joints + 0.01)
    assert bool(success.all())
    torch.testing.assert_close(
        solver.get_fk(solutions[:, 0]), targets, atol=2e-4, rtol=0.0
    )
