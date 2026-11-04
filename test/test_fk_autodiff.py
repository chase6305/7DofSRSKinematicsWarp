"""Separately compiled FK kernels retain Warp Tape gradients."""

import numpy as np
import pytest
import torch
import warp as wp

from srs_solver import SRSSolver, SRSSolverCfg
from warp_kernel import compute_fk_kernel

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_fk_tape_gradient_matches_analytical_linear_jacobian(device):
    cfg = SRSSolverCfg()
    cfg.T_b_ob[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cfg.T_b_ob[:3, 3] = [0.2, -0.1, 0.3]
    cfg.tcp[:3, 3] = [0.06, -0.02, 0.08]
    cfg.rotation_directions = np.array([-1, 1, -1, 1, -1, 1, -1])
    solver = SRSSolver(cfg, 2, device, use_warp=True)
    q = np.array(
        [[0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], [0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    joints = wp.array(q.ravel(), dtype=float, device=device, requires_grad=True)
    poses = wp.empty(2, dtype=wp.mat44, device=device, requires_grad=True)
    flags = wp.empty(2, dtype=int, device=device)
    seed = np.zeros((2, 4, 4), dtype=np.float32)
    seed[:, :3, 3] = [1, 2, -0.5]
    stream = wp.stream_from_torch(device) if device == "cuda" else None
    with wp.ScopedStream(stream, sync_exit=True):
        with wp.Tape() as tape:
            wp.launch(
                compute_fk_kernel,
                dim=2,
                inputs=[
                    joints,
                    solver.impl.dh_params_wp,
                    solver.impl.rotation_directions_wp,
                    wp.mat44(*cfg.T_b_ob.ravel()),
                    wp.mat44(*cfg.T_e_oe.ravel()),
                    wp.mat44(*cfg.tcp.ravel()),
                ],
                outputs=[poses, flags],
                device=device,
            )
        tape.backward(grads={poses: wp.array(seed, dtype=wp.mat44, device=device)})
        actual = wp.to_torch(joints.grad).view(2, 7).clone()
    jacobian = solver.get_jacobian(q)
    expected = torch.einsum(
        "c,ncj->nj", torch.tensor([1, 2, -0.5], device=device), jacobian[:, :3]
    )
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [1, 3, 8])
def test_unrolled_srs_path_preserves_generic_joint_counts(device, count):
    rng = np.random.default_rng(6305)
    q = rng.uniform(-1, 1, (2, count)).astype(np.float32)
    dh = rng.uniform(-0.2, 0.2, (count, 4)).astype(np.float32)
    directions = np.resize(np.array([-1, 1], dtype=np.float32), count)
    expected = []
    for row in q:
        pose = np.eye(4)
        for angle, direction, (d, alpha, a, offset) in zip(row, directions, dh):
            ct, st = np.cos(float(angle) * direction + offset), np.sin(
                float(angle) * direction + offset
            )
            ca, sa = np.cos(float(alpha)), np.sin(float(alpha))
            pose = pose @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
        expected.append(pose)
    output = wp.empty(2, dtype=wp.mat44, device=device)
    flags = wp.empty(2, dtype=int, device=device)
    identity = wp.mat44(*np.eye(4).ravel())
    wp.launch(
        compute_fk_kernel,
        dim=2,
        inputs=[
            wp.array(q.ravel(), dtype=float, device=device),
            wp.array(dh.ravel(), dtype=float, device=device),
            wp.array(directions, dtype=float, device=device),
            identity,
            identity,
            identity,
        ],
        outputs=[output, flags],
        device=device,
    )
    assert flags.numpy().tolist() == [1, 1]
    np.testing.assert_allclose(output.numpy(), expected, atol=2e-6, rtol=0)
