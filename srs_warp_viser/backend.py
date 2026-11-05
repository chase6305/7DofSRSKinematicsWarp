"""Torch/Warp adapter used by the shared SRS experiment model."""

import numpy as np

from srs_solver import KUKAiiwaSolver, SRSConfiguration


class Backend:
    def __init__(self, kind="warp", device="cpu"):
        if kind not in ("numpy", "warp"):
            raise ValueError("backend must be numpy or warp")
        self.solver = KUKAiiwaSolver(device=device, use_warp=kind == "warp")
        limits = self.solver.get_position_limits()
        self.lower, self.upper = limits["lower"], limits["upper"]
        self.urdf_path = self.solver.urdf_path
        self.label = (
            f"{'Warp' if kind == 'warp' else 'NumPy batch'} / {self.solver.device}"
        )

    def fk(self, q):
        return self.solver.get_fk(q)[0].detach().cpu().numpy().astype(float)

    def chains(self, joints):
        return self.solver.get_all_fk_mat(joints).detach().cpu().numpy().astype(float)

    def jacobian(self, q):
        return self.solver.get_jacobian(q).detach().cpu().numpy().astype(float)

    def configuration(self, q):
        config = self.solver.get_configuration(q)
        return (config.shoulder, config.elbow, config.wrist), config.redundancy

    def solve(self, target, branch, psi, seed):
        ok, joints = self.solver.solve_configuration(
            target, SRSConfiguration(*branch, float(psi)), seed
        )
        return joints.detach().cpu().numpy().astype(float) if ok else None

    def ik(self, target, seed):
        success, joints = self.solver.get_ik(target, np.asarray(seed))
        return (
            joints[0, 0].detach().cpu().numpy().astype(float)
            if bool(success[0])
            else None
        )

    def solve_many(self, target, configurations, seed):
        requests = tuple((tuple(branch), float(psi)) for branch, psi in configurations)
        cacheable = all(type(sign) is int for branch, _ in requests for sign in branch)
        cached = getattr(self, "_configuration_cache", None)
        if cacheable and cached is not None and requests == cached[0]:
            compiled = cached[1]
        else:
            compiled = tuple(SRSConfiguration(*branch, psi) for branch, psi in requests)
            self._configuration_cache = (requests, compiled) if cacheable else None
        success, joints = self.solver.solve_configurations(
            target,
            compiled,
            seed,
        )
        return success.cpu().numpy(), joints.cpu().numpy().astype(float)
