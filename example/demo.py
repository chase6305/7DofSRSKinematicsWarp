"""Repository demo entrypoint.

Examples:
    python demo.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from viser_viewer import SingleArmViserViewer

if TYPE_CHECKING:
    from srs_solver import SRSSolverCfg


@contextlib.contextmanager
def suppress_native_output():
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def build_solver_cfg() -> SRSSolverCfg:
    from srs_solver import SRSSolverCfg

    cfg = SRSSolverCfg()
    cfg.dh_params = np.array(
        [
            [0.34, -np.pi / 2.0, 0.0, 0.0],
            [0.0, np.pi / 2.0, 0.0, 0.0],
            [0.4, np.pi / 2.0, 0.0, 0.0],
            [0.0, -np.pi / 2.0, 0.0, 0.0],
            [0.4, -np.pi / 2.0, 0.0, 0.0],
            [0.0, np.pi / 2.0, 0.0, 0.0],
            [0.126, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cfg.link_lengths = np.array([0.34, 0.4, 0.4, 0.126], dtype=np.float64)
    cfg.rotation_directions = [1, 1, 1, 1, 1, 1, 1]
    cfg.T_b_ob = np.eye(4)
    cfg.T_e_oe = np.eye(4)
    cfg.num_samples = 300
    cfg.sort_ik = True
    cfg.ik_nearest_weight = [1, 1, 1, 1, 1, 1, 1]
    cfg.joint_names = [f"joint{i + 1}" for i in range(7)]
    cfg.urdf_path = str(
        Path(__file__).resolve().parent.parent / "urdf" / "lbr_iiwa7_r800.urdf"
    )
    return cfg


def _project_to_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rot)
    rot_proj = u @ vt
    if np.linalg.det(rot_proj) < 0.0:
        u[:, -1] *= -1.0
        rot_proj = u @ vt
    return rot_proj


def _interpolate_pose_linear(
    pose_a: np.ndarray, pose_b: np.ndarray, alpha: float
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
    rot_linear = (1.0 - alpha) * pose_a[:3, :3] + alpha * pose_b[:3, :3]
    pose[:3, :3] = _project_to_rotation_matrix(rot_linear)
    return pose


def _solve_ik_trajectory(
    solver, target_poses: list[np.ndarray], q_seed: np.ndarray
) -> list[np.ndarray]:
    trajectory_q: list[np.ndarray] = []
    seed = q_seed.astype(np.float32).reshape(1, 7)

    for step_index, target_pose in enumerate(target_poses):
        target_xpos = torch.from_numpy(target_pose).float().unsqueeze(0)
        qpos_seed = torch.from_numpy(seed)
        success, qpos = solver.get_ik(
            target_xpos, qpos_seed, return_all_solutions=False
        )
        success_np = bool(success.detach().cpu().numpy()[0])
        if not success_np:
            # Return the reachable prefix instead of crashing the demo.
            print(
                f"[traj] IK failed at step {step_index}, truncating to reachable prefix."
            )
            break

        q_np = qpos.detach().cpu().numpy().reshape(-1, 7)[0].astype(np.float64)
        trajectory_q.append(q_np)
        seed = q_np.astype(np.float32).reshape(1, 7)

    return trajectory_q


def _is_pose_reachable(
    solver, pose: np.ndarray, q_seed: np.ndarray
) -> tuple[bool, np.ndarray]:
    target_xpos = torch.from_numpy(pose).float().unsqueeze(0)
    seed = torch.from_numpy(q_seed.astype(np.float32).reshape(1, 7))
    success, qpos = solver.get_ik(target_xpos, seed, return_all_solutions=False)
    if not bool(success.detach().cpu().numpy()[0]):
        return False, q_seed
    q_np = qpos.detach().cpu().numpy().reshape(-1, 7)[0].astype(np.float64)
    return True, q_np


def _find_reachable_pose_b_with_fixed_orientation(
    solver,
    pose_a: np.ndarray,
    pose_b_raw: np.ndarray,
    q_seed: np.ndarray,
    max_checks: int = 40,
) -> tuple[np.ndarray, float]:
    pose_b_fixed = np.eye(4, dtype=np.float64)
    pose_b_fixed[:3, :3] = pose_a[:3, :3]

    p_a = pose_a[:3, 3]
    p_b = pose_b_raw[:3, 3]

    best_alpha = 0.0
    best_pose = pose_a.copy()
    best_q = q_seed.copy()

    for alpha in np.linspace(1.0, 0.0, max_checks):
        candidate = pose_b_fixed.copy()
        candidate[:3, 3] = (1.0 - alpha) * p_a + alpha * p_b
        reachable, q_next = _is_pose_reachable(solver, candidate, best_q)
        if reachable:
            best_alpha = float(alpha)
            best_pose = candidate
            best_q = q_next
            break

    return best_pose, best_alpha


def run_ab_pose_interp_ik_sim(host: str, port: int, steps: int, dt: float) -> None:
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5)

    with suppress_native_output():
        from srs_solver import SRSSolver

    cfg = build_solver_cfg()
    with suppress_native_output():
        solver = SRSSolver(cfg, num_envs=1, device="cpu")

    q_a = np.array(
        [np.pi / 6, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=np.float64
    )
    q_b = np.array(
        [-np.pi / 6, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0], dtype=np.float64
    )

    pose_a = solver.impl._get_fk(q_a)
    pose_b_raw = solver.impl._get_fk(q_b)
    pose_b, reachable_alpha = _find_reachable_pose_b_with_fixed_orientation(
        solver,
        pose_a,
        pose_b_raw,
        q_seed=q_a,
    )

    steps = max(2, int(steps))
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    target_poses = [
        _interpolate_pose_linear(pose_a, pose_b, float(alpha)) for alpha in alphas
    ]
    trajectory_q = _solve_ik_trajectory(solver, target_poses, q_seed=q_a)
    if len(trajectory_q) == 0:
        raise RuntimeError(
            "No reachable trajectory points were found under fixed orientation."
        )

    print("=== A/B pose interpolation IK demo ===")
    print("A joint (rad):", q_a)
    print("B joint (rad):", q_b)
    print("Interpolation steps:", steps)
    print("Reachable B ratio under fixed orientation:", f"{reachable_alpha:.3f}")

    viewer = SingleArmViserViewer(
        Path(__file__).resolve().parent.parent / "urdf" / "lbr_iiwa7_r800.urdf",
        host=host,
        port=port,
    )

    try:
        while True:
            for q in trajectory_q:
                viewer.set_qpos(q)
                time.sleep(dt)
            for q in reversed(trajectory_q):
                viewer.set_qpos(q)
                time.sleep(dt)
    except KeyboardInterrupt:
        pass

    print(f"Viewer ready at http://{host}:{port}. Press Ctrl+C to exit.")
    viewer.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo entrypoint for viewer and IK examples."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--steps", type=int, default=100, help="Interpolation steps for traj mode."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Execution period in seconds for traj mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ab_pose_interp_ik_sim(
        host=args.host, port=args.port, steps=args.steps, dt=args.dt
    )


if __name__ == "__main__":
    main()
