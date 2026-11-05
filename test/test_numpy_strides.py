"""Public tensor APIs accept reversed NumPy views without changing input arrays."""

import numpy as np
import pytest
import torch

from srs_solver import SRSSolver, SRSSolverCfg

BACKENDS = [(False, "cpu"), (True, "cpu")] + (
    [(True, "cuda")] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize("use_warp,device", BACKENDS)
@pytest.mark.parametrize("layout", ["batch", "joints"])
def test_reversed_numpy_joint_target_seed_and_transform_views(use_warp, device, layout):
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 4, device, use_warp=use_warp)
    storage = (
        np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
        + np.linspace(-0.1, 0.1, 4)[:, None]
    )
    q = storage[::-1] if layout == "batch" else storage[:, ::-1]
    assert any(stride < 0 for stride in q.strides)
    preserved = q.copy()
    for method in (solver.get_fk, solver.get_all_fk_mat, solver.get_jacobian):
        torch.testing.assert_close(method(q), method(q.copy()), atol=0, rtol=0)
    targets = solver.get_fk(q.copy()).cpu().numpy()[::-1]
    seeds = (q.copy() + 0.02)[::-1]
    for mode in ("global", "interval"):
        expected = solver.get_ik(targets.copy(), seeds.copy(), search_mode=mode)
        actual = solver.get_ik(targets, seeds, search_mode=mode)
        assert bool(actual[0].all())
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)

    single = q.copy()[0, ::-1]
    configuration = solver.get_configuration(single)
    target = solver.get_fk(single)[0].cpu().numpy()
    ok, joints = solver.solve_configuration(target, configuration, single)
    assert ok
    torch.testing.assert_close(
        solver.get_fk(joints)[0], torch.tensor(target, device=device), atol=2e-5, rtol=0
    )
    velocity = solver.get_null_space_velocity(single, np.linspace(-1, 1, 7)[::-1])
    assert (
        float(torch.linalg.vector_norm(solver.get_jacobian(single) @ velocity)) < 1e-5
    )

    # This reversed view is a valid identity transform, despite its layout.
    tcp = np.eye(4)[::-1].copy()[::-1]
    assert tcp.strides[0] < 0
    solver.set_tcp(tcp)
    np.testing.assert_array_equal(solver.get_tcp(), np.eye(4))
    np.testing.assert_array_equal(q, preserved)
    with pytest.raises(ValueError, match="qpos.*finite"):
        solver.get_fk(np.full((4, 7), np.nan)[::-1])
