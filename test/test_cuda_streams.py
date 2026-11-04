"""Torch consumers must observe Warp outputs on the caller's CUDA stream."""

import numpy as np
import pytest
import torch
import warp as wp

from srs_solver import SRSSolver, SRSSolverCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_torch_stream_orders_launches_outputs_and_cached_workspace(monkeypatch):
    reference = SRSSolver(SRSSolverCfg(num_samples=32), 1, "cpu", use_warp=False)
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    expected_fk = reference.get_fk(q)
    expected_chain = reference.get_all_fk_mat(q)
    expected_jacobian = reference.get_jacobian(q)
    saved_stream = wp.get_stream("cuda:0")
    launch = wp.launch
    launched = []

    def check_stream(*args, **kwargs):
        actual = wp.get_stream(kwargs["device"])
        assert actual.cuda_stream == torch.cuda.current_stream().cuda_stream
        launched.append(actual.cuda_stream)
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", check_stream)
    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.default_stream()]
    preserved = []
    for index, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            if index == 0:
                solver = SRSSolver(
                    SRSSolverCfg(num_samples=32), 1, "cuda", use_warp=True
                )
            joints = torch.tensor(q, dtype=torch.float32, device="cuda")
            # Clone immediately: no explicit synchronization between producer and consumer.
            fk = solver.get_fk(joints).clone()
            chain = solver.get_all_fk_mat(joints).clone()
            jacobian = solver.get_jacobian(joints).clone()
            ok, solutions = solver.get_ik(
                fk, joints + 0.02, return_all_solutions=index == 1
            )
            solved_pose = solver.get_fk(solutions[:, 0]).clone()
            preserved.append((fk, chain, jacobian, solved_pose, ok.clone()))
        assert wp.get_stream("cuda:0") == saved_stream

    for fk, chain, jacobian, solved_pose, ok in preserved:
        assert bool(ok.all())
        torch.testing.assert_close(fk.cpu(), expected_fk, atol=5e-7, rtol=0)
        torch.testing.assert_close(chain.cpu(), expected_chain, atol=5e-7, rtol=0)
        torch.testing.assert_close(jacobian.cpu(), expected_jacobian, atol=5e-7, rtol=0)
        torch.testing.assert_close(solved_pose.cpu(), expected_fk, atol=4e-6, rtol=0)
    assert len(launched) >= 18
