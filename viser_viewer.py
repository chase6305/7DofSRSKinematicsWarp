"""Standalone single-arm Viser viewer for the KUKA iiwa 7-DOF model.

This script runs a single built-in sinusoidal trajectory mode:

- loads the single-arm URDF in this repository
- animates all joints continuously in one main loop
- updates a TCP frame from the URDF FK so pose stays consistent with the arm

Dependencies:
- viser==1.0.24
- yourdfpy==0.0.58

Run:
        python viser_viewer.py
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np


class SingleArmViserViewer:
    def __init__(self, urdf_path: Path, host: str, port: int) -> None:
        np.set_printoptions(precision=5, suppress=True, linewidth=160)

        try:
            viser = importlib.import_module("viser")
            yourdfpy = importlib.import_module("yourdfpy")
            ViserUrdf = importlib.import_module("viser.extras").ViserUrdf
        except Exception as exc:
            raise ImportError(
                "Missing dependencies. Install viser and yourdfpy before running this viewer."
            ) from exc

        self._viser = viser
        self._yourdfpy = yourdfpy
        self._ViserUrdf = ViserUrdf
        self.host = host
        self.port = port

        self.urdf_path = urdf_path.resolve()
        self.server = viser.ViserServer(host=host, port=port)

        self.robot_model = yourdfpy.URDF.load(str(self.urdf_path))
        self.viser_robot = ViserUrdf(
            self.server,
            self.robot_model,
            root_node_name="/robot",
        )

        self.joint_names = list(self.viser_robot.get_actuated_joint_names())

        self.end_effector_link_name = self.robot_model.joint_map[
            self.joint_names[-1]
        ].child
        self.end_effector_link = self.robot_model.link_map.get(
            self.end_effector_link_name
        )

        self.q = np.zeros(len(self.joint_names), dtype=np.float64)
        self.viser_robot.update_cfg(self.q)

        self.server.scene.add_frame(
            "/world",
            axes_length=0.2,
            axes_radius=0.01,
        )
        tcp_frame_name = self.viser_robot._joint_frames[-1].name + "/tcp"

        self.tcp_frame = self.server.scene.add_frame(
            tcp_frame_name,
            axes_length=0.14,
            axes_radius=0.008,
        )

    def set_qpos(self, q: np.ndarray) -> None:
        self.q = q
        self.viser_robot.update_cfg(self.q)

    def serve_forever(self) -> None:
        print(f"Viser viewer running at http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


def _parse_args() -> argparse.Namespace:
    from srs_solver import get_iiwa_urdf_path

    parser = argparse.ArgumentParser(
        description="Standalone Viser viewer for a single 7-DOF arm."
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(get_iiwa_urdf_path()),
        help="Path to the robot URDF.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Viser server host."
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument(
        "--hz", type=float, default=60.0, help="Trajectory update rate in Hz."
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    viewer = SingleArmViserViewer(args.urdf, host=args.host, port=args.port)
    viewer.serve_forever()


if __name__ == "__main__":
    main()
