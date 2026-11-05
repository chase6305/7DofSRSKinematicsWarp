"""Minimal URDF viewer used by the single-arm trajectory example."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


class SingleArmViserViewer:
    def __init__(self, urdf_path: Path, host: str, port: int) -> None:
        try:
            import viser
            import yourdfpy
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError(
                "Install the viewer extra: pip install -e '.[viewer]'"
            ) from exc

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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    viewer = SingleArmViserViewer(args.urdf, host=args.host, port=args.port)
    viewer.serve_forever()


if __name__ == "__main__":
    main()
