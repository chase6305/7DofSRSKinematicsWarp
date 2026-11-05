"""Shared geometry preserves the direct Warp kernel's candidate contracts."""

import numpy as np
import pytest
import torch
import warp as wp

from srs_solver import SRSSolver, SRSSolverCfg
from warp_kernel import compute_ik_kernel

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("samples", [2, 17, 73])
def test_prepared_matches_direct_candidates_with_frames_and_singularities(
    device, samples
):
    cfg = SRSSolverCfg(num_samples=samples, sort_ik=False)
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    cfg.dh_params[:, 3] = np.linspace(-0.15, 0.15, 7)
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.2, -0.1, 0.3]
    cfg.T_e_oe[:3, 3] = [0.03, 0, 0.04]
    cfg.tcp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cfg.tcp[:3, 3] = [0.06, -0.02, 0.08]
    cfg.ik_nearest_weight = np.array([1, 2, 0, 3, 1, 0, 4])
    solver = SRSSolver(cfg, 5, device, use_warp=True)
    joints = np.tile([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], (5, 1))
    for row, middle in ((1, 1), (2, 5)):
        joints[row, middle] = (
            -cfg.dh_params[middle, 3] * cfg.rotation_directions[middle]
        )
    joints[3, 3] = (1e-4 - cfg.dh_params[3, 3]) * cfg.rotation_directions[3]
    target = solver.get_fk(joints)
    target[4, 0, 3] = 10
    seed = torch.tensor(joints + 0.1, dtype=torch.float32, device=device)
    for weights in (np.array([1, 2, 0, 3, 1, 0, 4]), np.zeros(7)):
        cfg.ik_nearest_weight[:] = weights
        success, actual = solver.get_ik(target, seed, return_all_solutions=True)
        workspace = solver.impl._workspace
        flags = wp.to_torch(workspace["success"]).clone().bool()
        count = flags.numel()
        reference_q = wp.empty(count * 7, dtype=float, device=device)
        reference_ok = wp.empty(count, dtype=int, device=device)
        stream = wp.stream_from_torch(device) if device != "cpu" else None
        with wp.ScopedStream(stream, sync_exit=True):
            wp.launch(
                compute_ik_kernel,
                dim=count,
                inputs=[
                    workspace["xpos"],
                    solver.impl.elbow_angles_wp,
                    solver.impl.qpos_limits_wp,
                    solver.impl.configs_wp,
                    solver.impl.dh_params_wp,
                    solver.impl.link_lengths_wp,
                    solver.impl.rotation_directions_wp,
                    wp.from_torch(seed.flatten()),
                    solver.impl.nearest_weights_wp,
                ],
                outputs=[reference_ok, reference_q],
                device=device,
            )
            expected_flags = wp.to_torch(reference_ok).bool()
            expected = wp.to_torch(reference_q).view(5, 8 * samples, 7)
            expected = expected.masked_fill(~expected_flags.view(5, -1, 1), 0)
        torch.testing.assert_close(flags, expected_flags)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
        assert not bool(success[4]) and not bool(actual[4].any())
        seed = seed - 0.2
    # Reusing the same geometry buffer must overwrite a formerly valid target.
    target[:, 0, 3] = 10
    success, solutions = solver.get_ik(target, seed, return_all_solutions=True)
    assert not bool(success.any()) and not bool(solutions.any())
