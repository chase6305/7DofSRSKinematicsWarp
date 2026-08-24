#!/usr/bin/env python3
"""Parallel Jacobian and fixed-TCP null-space direction demo."""

from __future__ import annotations

import argparse
import time

import torch

from srs_solver import KUKAiiwaSolver


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--backend", choices=("auto", "warp", "numpy"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch-size and repeats must be positive")

    device = torch.device(args.device)
    use_warp = None if args.backend == "auto" else args.backend == "warp"
    solver = KUKAiiwaSolver(num_envs=args.batch_size, device=device, use_warp=use_warp)
    limits = solver.get_position_limits()
    lower = torch.as_tensor(limits["lower"], dtype=torch.float32, device=device)
    upper = torch.as_tensor(limits["upper"], dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(11)
    joints = (
        lower * 0.6
        + torch.rand((args.batch_size, 7), generator=generator, device=device)
        * (upper - lower)
        * 0.6
    )

    directions = solver.get_null_space_direction(joints)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(args.repeats):
        directions = solver.get_null_space_direction(joints)
    synchronize(device)
    elapsed = (time.perf_counter() - started) / args.repeats

    jacobians = solver.get_jacobian(joints)
    residual = torch.linalg.vector_norm(
        (jacobians @ directions.unsqueeze(-1)).squeeze(-1), dim=1
    )
    norms = torch.linalg.vector_norm(directions, dim=1)
    print(
        f"device={device} backend={'warp' if solver.use_warp else 'numpy'} "
        f"batch={args.batch_size}"
    )
    print(
        f"Jacobian + null direction: {elapsed * 1e3:.3f} ms, {args.batch_size / elapsed:,.0f} states/s"
    )
    print(
        f"max ||J n||={float(residual.max()):.3e}, max | ||n||-1 |={float((norms - 1.0).abs().max()):.3e}"
    )


if __name__ == "__main__":
    main()
