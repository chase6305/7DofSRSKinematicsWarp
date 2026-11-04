"""Torch aliases must resolve to the same device used by Warp allocations."""

import numpy as np
import pytest
import torch

from srs_solver import SRSSolver, SRSSolverCfg, standardize_device_string


def test_cuda_alias_uses_current_device_without_overriding_explicit_indices(
    monkeypatch,
):
    # Alias resolution itself needs no GPU. Exercise a nonzero ordinal on CI.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert standardize_device_string("cuda") == "cuda:2"
    assert standardize_device_string(torch.device("cuda")) == "cuda:2"
    assert standardize_device_string("cuda:0") == "cuda:0"
    assert standardize_device_string(torch.device("cuda:1")) == "cuda:1"


@pytest.mark.parametrize("use_warp", [False, True])
def test_indexed_cpu_alias_round_trip(use_warp):
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 1, "cpu:0", use_warp=use_warp)
    assert solver.device == torch.device("cpu")
    assert standardize_device_string("cpu:0") == "cpu"
    q = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2])
    target = solver.get_fk(q)
    ok, joints = solver.get_ik(target, q + 0.02)
    assert bool(ok.all())
    torch.testing.assert_close(solver.get_fk(joints[:, 0]), target, atol=2e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_alias_is_fixed_when_solver_is_created(monkeypatch):
    current = torch.cuda.current_device()
    solver = SRSSolver(SRSSolverCfg(num_samples=17), 1, "cuda", use_warp=True)
    assert solver.device == torch.device("cuda", current)
    # Later alias resolution must not move this solver's buffers or outputs.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current + 1)
    q = torch.tensor([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], device=solver.device)
    target = solver.get_fk(q)
    ok, joints = solver.get_ik(target, q + 0.02)
    assert joints.device == solver.device and bool(ok.all())
    torch.testing.assert_close(solver.get_fk(joints[:, 0]), target, atol=2e-5, rtol=0)
