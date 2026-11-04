#!/usr/bin/env python3
"""Measure first FK/IK calls with an empty Warp cache, then warmed IK latency."""

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=73)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_samples < 2 or args.repeats < 1:
        parser.error("batch-size/repeats must be positive and num-samples must be >= 2")
    if args.output and args.output.exists():
        parser.error("output file already exists")
    # Set before importing CUDA libraries; only this benchmark process changes.
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import warp as wp

    from srs_solver import SRSSolver, SRSSolverCfg

    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with tempfile.TemporaryDirectory(prefix="srs-startup-") as cache:
        wp.config.kernel_cache_dir = cache
        with redirect_stdout(sys.stderr):
            started = time.perf_counter()
            solver = SRSSolver(
                SRSSolverCfg(num_samples=args.num_samples),
                args.batch_size,
                args.device,
                use_warp=True,
            )
            q = torch.tensor(
                [0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], device=solver.device
            ).repeat(args.batch_size, 1)
            target = solver.get_fk(q)
            synchronize(solver.device)
            first_fk_ms = (time.perf_counter() - started) * 1000
            seed = q + 0.02
            started = time.perf_counter()
            success, joints = solver.get_ik(target, seed, search_mode="global")
            synchronize(solver.device)
            first_ik_ms = (time.perf_counter() - started) * 1000
            actual = solver.get_fk(joints[:, 0])
            position = torch.linalg.vector_norm(
                actual[:, :3, 3] - target[:, :3, 3], dim=1
            ).max()
            rotation = (
                2
                * torch.asin(
                    torch.clamp(
                        torch.linalg.matrix_norm(actual[:, :3, :3] - target[:, :3, :3])
                        / (8**0.5),
                        0,
                        1,
                    )
                )
            ).max()
            if (
                not bool(success.all())
                or float(position) > 2e-4
                or float(rotation) > 2e-4
            ):
                raise AssertionError("startup IK failed or exceeded TCP tolerance")
            for _ in range(5):
                solver.get_ik(target, seed, search_mode="global")
            synchronize(solver.device)
            elapsed = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                solver.get_ik(target, seed, search_mode="global")
                synchronize(solver.device)
                elapsed.append((time.perf_counter() - started) * 1000)
        report = {
            "device": str(solver.device),
            "environment": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "torch", "warp-lang")},
            },
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "driver_cache_disabled": True,
            "scope": "Empty temporary Warp cache; imports excluded. Constructor/first FK includes device initialization. First IK includes compilation still required after FK. Warmed calls synchronize on CUDA. No viewer.",
            "constructor_first_fk_ms": first_fk_ms,
            "additional_first_ik_ms": first_ik_ms,
            "warmed_ik_median_ms": statistics.median(elapsed),
            "warmed_ik_samples_ms": elapsed,
            "max_position_error_m": float(position),
            "max_rotation_error_rad": float(rotation),
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
