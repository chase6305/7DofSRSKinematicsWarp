import numpy as np
import warp as wp
from warp_kernel import compute_fk_kernel, compute_ik_kernel


def dh_transform(d, alpha, a, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_warp_fk_kernel_matches_dh_with_nonzero_a(tmp_path):
    # A writable cache makes this test work in containers with a read-only home.
    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    dh = np.array(
        [
            [0.1, -0.2, 0.03, 0.0],
            [0.0, 0.3, 0.04, 0.1],
            [0.2, -0.4, 0.02, 0.0],
            [0.0, 0.2, 0.05, -0.1],
            [0.1, -0.1, 0.01, 0.0],
            [0.0, 0.4, 0.02, 0.2],
            [0.08, 0.0, 0.06, 0.0],
        ],
        dtype=np.float32,
    )
    joints = np.array([[0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]], dtype=np.float32)
    poses = wp.empty(1, dtype=wp.mat44, device="cpu")
    success = wp.empty(1, dtype=int, device="cpu")
    identity = wp.mat44(*np.eye(4, dtype=np.float32).ravel())
    wp.launch(
        compute_fk_kernel,
        dim=1,
        inputs=[
            wp.array(joints.ravel(), dtype=float, device="cpu"),
            wp.array(dh.ravel(), dtype=float, device="cpu"),
            wp.array(np.ones(7, dtype=np.float32), dtype=float, device="cpu"),
            identity,
            identity,
            identity,
        ],
        outputs=[poses, success],
        device="cpu",
    )
    expected = np.eye(4, dtype=np.float32)
    for row, joint in zip(dh, joints[0]):
        expected = expected @ dh_transform(row[0], row[1], row[2], row[3] + joint)
    np.testing.assert_allclose(poses.numpy()[0], expected, atol=2e-6)
    assert success.numpy()[0] == 1


def test_warp_parallel_ik_round_trip_on_cpu(tmp_path):
    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    half_pi = np.pi / 2.0
    dh = np.array(
        [
            [0.34, -half_pi, 0.0, 0.0],
            [0.0, half_pi, 0.0, 0.0],
            [0.4, half_pi, 0.0, 0.0],
            [0.0, -half_pi, 0.0, 0.0],
            [0.4, -half_pi, 0.0, 0.0],
            [0.0, half_pi, 0.0, 0.0],
            [0.126, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    known = np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2], dtype=np.float32)
    target = np.eye(4, dtype=np.float32)
    for row, joint in zip(dh, known):
        target = target @ dh_transform(row[0], row[1], row[2], row[3] + joint)
    configs = np.array(
        [[s, e, w] for s in (1.0, -1.0) for e in (1.0, -1.0) for w in (1.0, -1.0)],
        dtype=np.float32,
    )
    angles = np.linspace(-np.pi, np.pi, 73, dtype=np.float32)
    count = 8 * 73
    success = wp.zeros(count, dtype=int, device="cpu")
    qpos = wp.zeros(count * 7, dtype=float, device="cpu")
    wp.launch(
        compute_ik_kernel,
        dim=count,
        inputs=[
            wp.array([target], dtype=wp.mat44, device="cpu"),
            wp.array(angles, dtype=float, device="cpu"),
            wp.array(np.tile([-np.pi, np.pi], (7, 1)), dtype=wp.vec2, device="cpu"),
            wp.array(configs, dtype=wp.vec3, device="cpu"),
            wp.array(dh.ravel(), dtype=float, device="cpu"),
            wp.array([0.34, 0.4, 0.4, 0.126], dtype=float, device="cpu"),
            wp.array(np.ones(7, dtype=np.float32), dtype=float, device="cpu"),
            wp.zeros(7, dtype=float, device="cpu"),
            wp.array(np.ones(7, dtype=np.float32), dtype=float, device="cpu"),
        ],
        outputs=[success, qpos],
        device="cpu",
    )
    assert np.count_nonzero(success.numpy()) > 0
