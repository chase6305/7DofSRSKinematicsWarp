"""FK rejects nonfinite input on the solver stream and recovers on the next call."""

import contextlib
import gc

import numpy as np
import pytest
import torch
import warp as wp

from srs_solver import SRSSolver, SRSSolverCfg
from warp_kernel import compute_fk_kernel

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_fk_rejected_strided_batch_recovers_on_current_stream(device, value):
    solver = SRSSolver(SRSSolverCfg(), 3, device, use_warp=True)
    joints = torch.tensor(
        np.tile([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], (6, 1)),
        dtype=torch.float32,
        device=device,
    )[::2]
    assert not joints.is_contiguous()
    expected = solver.get_fk(joints)
    chain = solver.get_all_fk_mat(joints)
    stream = torch.cuda.Stream() if device == "cuda" else None
    scope = (
        torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    )
    with scope:
        bad = joints.clone()
        bad[1, 6] = value
        with pytest.raises(ValueError, match="qpos.*finite"):
            solver.get_fk(bad)
        empty = solver.get_fk(torch.empty((0, 7), device=device))
        assert empty.shape == (0, 4, 4)
        recovered = solver.get_fk(joints).clone()
    if stream is not None:
        stream.synchronize()
    del solver
    gc.collect()
    torch.testing.assert_close(recovered, expected, atol=0, rtol=0)
    torch.testing.assert_close(chain[:, -1], expected, atol=5e-7, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_direct_fk_kernel_reports_invalid_rows_and_zeroes_their_poses(device):
    solver = SRSSolver(SRSSolverCfg(), 3, device, use_warp=True)
    joints = torch.zeros((3, 7), device=device)
    joints[1, 0] = torch.nan
    joints[2, 6] = -torch.inf
    output = wp.empty(3, dtype=wp.mat44, device=device)
    success = wp.empty(3, dtype=int, device=device)
    stream = wp.stream_from_torch(device) if device == "cuda" else None
    with wp.ScopedStream(stream, sync_exit=True):
        wp.launch(
            compute_fk_kernel,
            dim=3,
            inputs=[
                wp.from_torch(joints.flatten()),
                solver.impl.dh_params_wp,
                solver.impl.rotation_directions_wp,
                wp.mat44(*solver.cfg.T_b_ob.ravel()),
                wp.mat44(*solver.cfg.T_e_oe.ravel()),
                wp.mat44(*solver.cfg.tcp.ravel()),
            ],
            outputs=[output, success],
            device=device,
        )
        flags = wp.to_torch(success).clone()
        poses = wp.to_torch(output).clone()
    assert flags.tolist() == [1, 0, 0]
    assert not bool(poses[1:].any())
    torch.testing.assert_close(poses[:1], solver.get_fk(torch.zeros(7)), atol=0, rtol=0)
