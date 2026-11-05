"""Check the displayed model against independent URDF integration."""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from srs_solver import KUKAiiwaSolver, get_iiwa_urdf_path


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


@pytest.mark.parametrize("use_warp", [False, True])
def test_displayed_urdf_matches_solver_fk_and_limits(use_warp):
    path = Path(get_iiwa_urdf_path())
    robot = ElementTree.parse(path).getroot()
    joints = robot.findall("joint")
    actuated = [j for j in joints if j.get("type") == "revolute"]
    assert [j.get("name") for j in actuated] == [f"joint{i}" for i in range(1, 8)]
    solver = KUKAiiwaSolver(device="cpu", use_warp=use_warp)
    limits = solver.get_position_limits()
    lower = np.array([float(j.find("limit").get("lower")) for j in actuated])
    upper = np.array([float(j.find("limit").get("upper")) for j in actuated])
    np.testing.assert_allclose(lower, limits["lower"], atol=1e-7)
    np.testing.assert_allclose(upper, limits["upper"], atol=1e-7)
    samples = np.random.default_rng(14).uniform(lower, upper, (20, 7))
    actual = solver.get_fk(samples).cpu().numpy()
    for sample, expected in zip(samples, actual):
        pose, index = np.eye(4), 0
        for joint in joints:
            origin = joint.find("origin")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            offset = np.eye(4)
            offset[:3, :3] = (
                rotation([0, 0, 1], yaw)
                @ rotation([0, 1, 0], pitch)
                @ rotation([1, 0, 0], roll)
            )
            offset[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            pose = pose @ offset
            if joint.get("type") == "revolute":
                turn = np.eye(4)
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                turn[:3, :3] = rotation(axis, sample[index])
                pose = pose @ turn
                index += 1
        np.testing.assert_allclose(pose[:3, 3], expected[:3, 3], atol=2e-6)
        np.testing.assert_allclose(pose[:3, :3], expected[:3, :3], atol=2e-6)
    assert (path.parent / "ASSET_LICENSE").is_file()
    for mesh in robot.findall(".//mesh"):
        assert (path.parent / mesh.get("filename")).is_file()
    assert all(
        m.get("filename").endswith(".glb")
        for m in robot.findall(".//visual/geometry/mesh")
    )
