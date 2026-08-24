#!/usr/bin/env python3
"""Batched Warp FK/IK round-trip and throughput demo."""

from __future__ import annotations

import argparse
import time

import torch

from srs_solver import SRSSolver, SRSSolverCfg


def pose_errors(
    actual: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.linalg.vector_norm(actual[:, :3, 3] - target[:, :3, 3], dim=1)
    relative = actual[:, :3, :3].transpose(1, 2) @ target[:, :3, :3]
    cosine = (relative.diagonal(dim1=1, dim2=2).sum(1) - 1.0) * 0.5
    sine_vector = 0.5 * torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        dim=1,
    )
    sine = torch.linalg.vector_norm(sine_vector, dim=1)
    return position, torch.atan2(sine, cosine)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=73)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--backend", choices=("auto", "warp", "numpy"), default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.num_samples, args.repeats) < 1:
        parser.error("batch-size, num-samples and repeats must be positive")

    device = torch.device(args.device)
    use_warp = None if args.backend == "auto" else args.backend == "warp"
    cfg = SRSSolverCfg(num_samples=args.num_samples)
    solver = SRSSolver(
        cfg, num_envs=args.batch_size, device=str(device), use_warp=use_warp
    )

    generator = torch.Generator(device=device).manual_seed(7)
    lower = torch.as_tensor(cfg.qpos_limits[:, 0], dtype=torch.float32, device=device)
    upper = torch.as_tensor(cfg.qpos_limits[:, 1], dtype=torch.float32, device=device)
    # Stay away from singular straight-arm configurations and exact limits.
    joints = (
        lower * 0.7
        + torch.rand((args.batch_size, 7), generator=generator, device=device)
        * (upper - lower)
        * 0.7
    )
    elbow_sign = torch.where(joints[:, 3] < 0.0, -1.0, 1.0)
    joints[:, 3] = torch.where(
        joints[:, 3].abs() < 0.25, elbow_sign * 0.25, joints[:, 3]
    )
    seed = joints + 0.02 * torch.randn(
        (args.batch_size, 7), generator=generator, device=device
    )

    targets = solver.get_fk(joints)
    solver.get_ik(targets, seed)
    synchronize(device)

    started = time.perf_counter()
    for _ in range(args.repeats):
        targets = solver.get_fk(joints)
    synchronize(device)
    fk_seconds = (time.perf_counter() - started) / args.repeats

    started = time.perf_counter()
    for _ in range(args.repeats):
        success, solutions = solver.get_ik(targets, seed)
    synchronize(device)
    ik_seconds = (time.perf_counter() - started) / args.repeats

    solved = solutions.reshape(-1, 7)
    reconstructed = solver.get_fk(solved)
    position, rotation = pose_errors(reconstructed, targets)
    valid = success.bool()
    print(
        f"device={device} backend={'warp' if solver.use_warp else 'numpy'} "
        f"batch={args.batch_size} psi_samples={args.num_samples}"
    )
    print(f"FK: {fk_seconds * 1e3:.3f} ms, {args.batch_size / fk_seconds:,.0f} poses/s")
    print(
        f"IK: {ik_seconds * 1e3:.3f} ms, {args.batch_size / ik_seconds:,.0f} targets/s"
    )
    print(f"success: {int(valid.sum())}/{args.batch_size}")
    if valid.any():
        print(
            "max round-trip error: "
            f"{float(position[valid].max()):.3e} m, "
            f"{float(rotation[valid].max()):.3e} rad"
        )


if __name__ == "__main__":
    main()
