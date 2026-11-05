import logging
import tempfile
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import warp as wp
from warp_kernel import (
    IKGeometry,
    compute_fk_chain_kernel,
    compute_fk_kernel,
    compute_prepared_ik_kernel,
    compute_jacobian_kernel,
    ik_distance_kernel,
    prepare_ik_geometry_kernel,
    validate_ik_inputs_kernel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ArmKineParams",
    "get_iiwa_urdf_path",
    "KUKAiiwa7R800Solver",
    "KUKAiiwaSolver",
    "SRSConfiguration",
    "SRSSolver",
    "SRSSolverCfg",
]


def _ensure_writable_warp_cache() -> None:
    """Fall back to /tmp when the configured Warp cache is read-only."""
    if wp.config.kernel_cache_dir is None:
        fallback = Path(tempfile.gettempdir()) / "srs-kinematics-warp-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(fallback)
        return
    configured = Path(wp.config.kernel_cache_dir)
    try:
        configured.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=configured):
            pass
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "srs-kinematics-warp-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(fallback)


def _resolve_device(device: str | torch.device) -> torch.device:
    """Keep Torch and Warp on the same concrete device for the solver lifetime."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        return torch.device("cpu")
    return device


def standardize_device_string(device: str | torch.device) -> str:
    """Resolve Torch aliases, including the current unindexed CUDA device."""
    return str(_resolve_device(device))


def _torch_stream(device: torch.device):
    """Order shared Torch/Warp buffers on the caller's current CUDA stream.

    Join back to Warp's previous stream on exit so cached workspaces and Warp
    allocations remain ordered when the next call uses a different Torch stream.
    CPU execution needs no stream conversion.
    """
    stream = wp.stream_from_torch(device) if device.type == "cuda" else None
    return wp.ScopedStream(stream, sync_exit=True)


def _as_tensor(value, *args, **kwargs):
    """Accept reversed NumPy views that Torch cannot wrap directly."""
    if isinstance(value, np.ndarray) and any(stride < 0 for stride in value.strides):
        value = value.copy()
    return torch.as_tensor(value, *args, **kwargs)


def _torch_view(array):
    """Share Warp storage through DLPack with explicit array ownership."""
    return torch.from_dlpack(wp.to_dlpack(array))


def _validate_pose(pose, name="transform", tolerance=1e-6):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=tolerance, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=tolerance, rtol=0)
    ):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return pose


@dataclass
class ArmKineParams:
    """Kinematics parameters.

    - dh_params stored as numpy array of shape (7,4).
    - qpos_limits stored as numpy array of shape (7,2) in radians.
    """

    # (initialized in __post_init__)
    # physical constants
    d_list: list[float] = field(init=False, default_factory=list)
    link_lengths: list[float] = field(init=False, default_factory=list)
    rotation_directions: list[float] = field(init=False, default_factory=list)

    # transforms
    T_b_ob: np.ndarray = field(init=False)
    T_e_oe: np.ndarray = field(init=False)

    # kinematic parameters
    dh_params: np.ndarray = field(init=False)
    qpos_limits: np.ndarray = field(init=False)

    def __post_init__(self):
        self.link_lengths = np.array([0.34, 0.4, 0.4, 0.126])
        self.T_b_ob = np.eye(4)
        self.T_e_oe = np.eye(4)
        self.dh_params = np.array(
            [
                [self.link_lengths[0], -np.pi / 2, 0.0, 0.0],
                [0.0, np.pi / 2, 0.0, 0.0],
                [self.link_lengths[1], np.pi / 2, 0.0, 0.0],
                [0.0, -np.pi / 2, 0.0, 0.0],
                [self.link_lengths[2], -np.pi / 2, 0.0, 0.0],
                [0.0, np.pi / 2, 0.0, 0.0],
                [self.link_lengths[3], 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.qpos_limits = np.tile(np.deg2rad([-180.0, 180.0]), (7, 1))
        self.rotation_directions = np.ones(7, dtype=int)


@dataclass
class SRSSolverCfg:
    """Configuration for SRS inverse kinematics controller."""

    class_type: str = "SRSSolver"
    # SRS-specific parameters
    dh_params: list = field(default_factory=list)
    T_b_ob: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_e_oe: np.ndarray = field(default_factory=lambda: np.eye(4))
    link_lengths: list = field(default_factory=list)
    rotation_directions: list = field(default_factory=list)
    qpos_limits: np.ndarray | None = None
    tcp: np.ndarray = field(default_factory=lambda: np.eye(4))
    ik_nearest_weight: list = field(default_factory=lambda: [1.0] * 7)
    num_samples: int = 100
    sort_ik: bool = True

    def __post_init__(self):
        kine = ArmKineParams()
        if len(self.dh_params) == 0:
            self.dh_params = kine.dh_params
        if len(self.link_lengths) == 0:
            self.link_lengths = kine.link_lengths
        if len(self.rotation_directions) == 0:
            self.rotation_directions = kine.rotation_directions
        if self.qpos_limits is None:
            self.qpos_limits = kine.qpos_limits

        self.validate()

    def validate(self) -> None:
        """Normalize and validate fields, including user mutations after creation."""
        self.dh_params = np.asarray(self.dh_params, dtype=np.float64)
        self.link_lengths = np.asarray(self.link_lengths, dtype=np.float64)
        self.rotation_directions = np.asarray(
            self.rotation_directions, dtype=np.float64
        )
        self.qpos_limits = np.asarray(self.qpos_limits, dtype=np.float64)
        self.T_b_ob = np.asarray(self.T_b_ob, dtype=np.float64)
        self.T_e_oe = np.asarray(self.T_e_oe, dtype=np.float64)
        self.tcp = np.asarray(self.tcp, dtype=np.float64)
        expected_shapes = {
            "dh_params": (self.dh_params, (7, 4)),
            "link_lengths": (self.link_lengths, (4,)),
            "rotation_directions": (self.rotation_directions, (7,)),
            "qpos_limits": (self.qpos_limits, (7, 2)),
            "T_b_ob": (self.T_b_ob, (4, 4)),
            "T_e_oe": (self.T_e_oe, (4, 4)),
            "tcp": (self.tcp, (4, 4)),
        }
        for name, (value, shape) in expected_shapes.items():
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
        if np.any(self.qpos_limits[:, 0] > self.qpos_limits[:, 1]):
            raise ValueError("each lower joint limit must not exceed its upper limit")
        if (
            isinstance(self.num_samples, (bool, np.bool_))
            or not isinstance(self.num_samples, (int, np.integer))
            or self.num_samples < 2
        ):
            raise ValueError("num_samples must be an integer >= 2")
        if not np.all(
            (self.rotation_directions == 1) | (self.rotation_directions == -1)
        ):
            raise ValueError("rotation_directions must contain only -1 or 1")
        if np.any(self.link_lengths[1:3] <= 0):
            raise ValueError("upper-arm and forearm lengths must be positive")
        weights = np.asarray(self.ik_nearest_weight, dtype=float)
        if (
            weights.shape != (7,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "ik_nearest_weight must be seven finite non-negative values"
            )
        self.ik_nearest_weight = weights.copy()
        for name in ("T_b_ob", "T_e_oe", "tcp"):
            _validate_pose(getattr(self, name), name)


@dataclass(frozen=True)
class SRSConfiguration:
    """Discrete shoulder/elbow/wrist branch and continuous arm angle."""

    shoulder: int
    elbow: int
    wrist: int
    redundancy: float

    def __post_init__(self):
        for name in ("shoulder", "elbow", "wrist"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value not in (-1, 1)
            ):
                raise ValueError(f"{name} must be -1 or 1")
        if not isinstance(
            self.redundancy, (int, float, np.integer, np.floating)
        ) or not np.isfinite(self.redundancy):
            raise ValueError("redundancy must be finite")


class _BaseSRSSolverImpl:
    """Base implementation for the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.dofs = 7
        self.dh_params = np.asarray(cfg.dh_params)
        self._parse_params()

    def _parse_params(self):
        # Compute the inverse transformation matrices for TCP, end-effector, and base.
        self.tcp_xpos = self.cfg.tcp
        self.tcp_inv_np = np.linalg.inv(self.tcp_xpos)
        self.T_e_oe_inv_np = np.linalg.inv(self.cfg.T_e_oe)
        self.T_b_ob_inv_np = np.linalg.inv(self.cfg.T_b_ob)

        # Convert configuration parameters to numpy arrays for efficient computation.
        self.dh_params_np = np.asarray(self.cfg.dh_params)
        self.link_lengths_np = np.asarray(self.cfg.link_lengths)
        self.rotation_directions_np = np.asarray(self.cfg.rotation_directions)

        self.ik_nearst_weight = np.asarray(self.cfg.ik_nearest_weight, dtype=float)

    @property
    def _distance_weights(self):
        # A common positive scale does not affect nearest-solution ordering.
        # Normalize before squaring or converting to float32 to avoid overflow
        # and underflow for otherwise valid finite user weights.
        scale = np.max(self.ik_nearst_weight)
        return self.ik_nearst_weight / scale if scale > 0 else self.ik_nearst_weight

    def _refresh_weights(self):
        weights = np.asarray(self.cfg.ik_nearest_weight, dtype=float)
        if (
            weights.shape != (7,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "ik_nearest_weight must be seven finite non-negative values"
            )
        self.ik_nearst_weight = weights


class _CPUSRSSolverImpl(_BaseSRSSolverImpl):
    """NumPy batch fallback with the same geometry and limit rules as the reference."""

    def _parse_params(self):
        super()._parse_params()
        self.dof = 7
        self.link_lengths = self.link_lengths_np
        self.world_to_base = self.cfg.T_b_ob
        self.flange_to_ee = self.cfg.T_e_oe @ self.tcp_xpos
        self.configs = [
            np.array(
                [
                    1 if i & 1 == 0 else -1,
                    1 if i & 2 == 0 else -1,
                    1 if i & 4 == 0 else -1,
                ]
            )
            for i in range(8)
        ]
        self.elbow_angles = np.linspace(
            -np.pi, np.pi, self.cfg.num_samples, endpoint=False
        )

    def _validate_inputs(self, target_xpos, qpos_seed):
        """Keep rigid-pose checks in float32 inside an application autocast scope."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            if not bool(torch.isfinite(target_xpos).all()):
                raise ValueError("target_xpos must contain only finite values")
            rotation = target_xpos[:, :3, :3]
            identity = torch.eye(3, dtype=target_xpos.dtype, device=self.device)
            bottom = torch.tensor(
                [0, 0, 0, 1], dtype=target_xpos.dtype, device=self.device
            )
            if (
                not torch.allclose(
                    target_xpos[:, 3],
                    bottom,
                    atol=1e-6,
                    rtol=0,
                )
                or not torch.allclose(
                    rotation.transpose(1, 2) @ rotation,
                    identity,
                    atol=1e-6,
                    rtol=0,
                )
                or not torch.allclose(
                    torch.linalg.det(rotation),
                    target_xpos.new_tensor(1.0),
                    atol=1e-6,
                    rtol=0,
                )
            ):
                raise ValueError(
                    "target_xpos must contain rigid homogeneous transforms"
                )
            if qpos_seed is not None:
                if not bool(torch.isfinite(qpos_seed).all()):
                    raise ValueError("qpos_seed must contain only finite values")

    def _dh_transform(self, d, alpha, a, theta) -> np.ndarray:
        """Build standard DH transforms, broadcasting over joint arrays."""
        shape = np.broadcast(d, alpha, a, theta).shape
        return self._fill_dh(
            np.zeros(shape + (4, 4)),
            d,
            a,
            np.cos(theta),
            np.sin(theta),
            np.cos(alpha),
            np.sin(alpha),
        )

    @staticmethod
    def _fill_dh(transforms, d, a, ct, st, ca, sa):
        """Fill DH entries in zeroed output storage."""
        transforms[..., 0, 0] = ct
        transforms[..., 0, 1] = -st * ca
        transforms[..., 0, 2] = st * sa
        transforms[..., 0, 3] = a * ct
        transforms[..., 1, 0] = st
        transforms[..., 1, 1] = ct * ca
        transforms[..., 1, 2] = -ct * sa
        transforms[..., 1, 3] = a * st
        transforms[..., 2, 1] = sa
        transforms[..., 2, 2] = ca
        transforms[..., 2, 3] = d
        transforms[..., 3, 3] = 1.0
        return transforms

    def _iter_joint_transforms(self, qpos, stop=7):
        """Yield joint transforms; large batches reuse one matrix buffer."""
        if qpos.ndim != 2 or len(qpos) < 256:
            transforms = self._joint_transforms(qpos, stop)
            if qpos.ndim > 1:
                transforms = np.moveaxis(transforms, -3, 0)
            yield from transforms
            return
        d, alpha, a, theta = self.dh_params[:stop].T
        ct = qpos[:, :stop] + theta
        st = np.sin(ct)
        np.cos(ct, out=ct)
        ca, sa = np.cos(alpha), np.sin(alpha)
        transform = np.zeros((len(qpos), 4, 4))
        for index in range(stop):
            yield self._fill_dh(
                transform,
                d[index],
                a[index],
                ct[:, index],
                st[:, index],
                ca[index],
                sa[index],
            )

    def _joint_transforms(self, qpos, stop=7):
        d, alpha, a, theta = self.dh_params[:stop].T
        return self._dh_transform(d, alpha, a, theta + qpos[..., :stop])

    def _fk_chain(self, qpos, world=True):
        transform = self.world_to_base.copy() if world else np.eye(4)
        poses = []
        for joint_transform in self._joint_transforms(qpos):
            transform = transform @ joint_transform
            poses.append(transform)
        return poses

    def _configuration(self, rconf: int) -> tuple:
        """Decode shoulder, elbow and wrist signs from the three branch bits."""
        arm_config = -1 if rconf & 1 else 1
        elbow_config = -1 if rconf & 2 else 1
        wrist_config = -1 if rconf & 4 else 1
        return arm_config, elbow_config, wrist_config

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        """Return the cross-product matrix of a three-vector."""
        return np.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def _calculate_joint_angles(
        self, P_s_to_w: np.ndarray, elbow_GC4: int
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Build reference shoulder/elbow angles for a reachable wrist centre."""
        d_se, d_ew = self.link_lengths[1:3]
        joints = np.zeros(7)

        # Check reachability and calculate elbow joint angle
        norm_P26 = np.linalg.norm(P_s_to_w)
        if norm_P26 < 1e-12:
            return False, None
        # Float32 pose rounding can put full extension just outside the triangle.
        # Returned solutions still have to pass the FK residual checks.
        if not (d_se + d_ew + 1e-6 >= norm_P26 and norm_P26 + 1e-6 >= abs(d_se - d_ew)):
            logger.debug("Specified pose outside reachable workspace.")
            return False, None

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
        if abs(elbow_cos_angle) > 1.0 + 1e-6:
            return False, None

        joints[3] = elbow_GC4 * np.arccos(np.clip(elbow_cos_angle, -1.0, 1.0))

        # Calculate joint 1
        if np.hypot(P_s_to_w[0], P_s_to_w[1]) > 1e-10:
            joints[0] = np.arctan2(P_s_to_w[1], P_s_to_w[0])
        else:
            joints[0] = 0

        # Calculate joint 2
        euclidean_norm = np.hypot(P_s_to_w[0], P_s_to_w[1])
        angle_phi = np.arccos(
            np.clip(
                (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26), -1.0, 1.0
            )
        )
        joints[1] = np.arctan2(euclidean_norm, P_s_to_w[2]) + elbow_GC4 * angle_phi

        return True, joints

    def _reference_geometry(self, pose, elbow):
        """Return the reference shoulder rotation and physical joint angles."""
        wrist = pose[:3, 3] - pose[:3, 2] * self.dh_params[-1, 0]
        shoulder = np.array([0.0, 0.0, self.link_lengths[0]])
        ok, joints = self._calculate_joint_angles(wrist - shoulder, elbow)
        if not ok:
            return None, None
        transforms = self._joint_transforms(joints - self.dh_params[:, 3], stop=3)
        rotation = transforms[0, :3, :3] @ transforms[1, :3, :3] @ transforms[2, :3, :3]
        return rotation, joints

    def _prepare_ik(self, local_target, elbow):
        """Reuse geometry for the last local target, with at most two elbow branches.

        Content keys detect in-place target or calibration edits. Joint seeds,
        limits and weights are deliberately excluded: candidate decomposition and
        validation still use their current values on every call.
        """
        arrays = (
            np.asarray(value)
            for value in (local_target, self.dh_params, self.link_lengths)
        )
        key = tuple((value.shape, value.dtype.str, value.tobytes()) for value in arrays)
        cached = getattr(self, "_ik_geometry_cache", None)
        geometry = {} if cached is None or cached[0] != key else cached[1]
        if elbow not in geometry:
            prepared = self._build_ik_geometry(local_target, elbow)
            if prepared is not None:
                for matrix in (*prepared[0], *prepared[1]):
                    matrix.setflags(write=False)
            geometry[elbow] = prepared
        self._ik_geometry_cache = (key, geometry)
        return geometry[elbow]

    def _build_ik_geometry(self, local_target, elbow):
        """Compute seed-independent shoulder and wrist coefficients."""
        reference, joints = self._reference_geometry(local_target, elbow)
        if reference is None:
            return None
        axis = local_target[:3, 3] - local_target[:3, 2] * self.dh_params[-1, 0]
        axis = axis - np.array([0.0, 0.0, self.link_lengths[0]])
        axis /= np.linalg.norm(axis)
        skew = self.skew(axis)
        shoulder = (
            skew @ reference,
            -skew @ skew @ reference,
            np.outer(axis, axis) @ reference,
        )
        d, alpha, a, _ = self.dh_params[3]
        elbow_rotation = self._dh_transform(d, alpha, a, joints[3])[:3, :3]
        wrist = tuple(
            elbow_rotation.T @ matrix.T @ local_target[:3, :3] for matrix in shoulder
        )
        return shoulder, wrist, joints[3]

    def _candidate_joint_angles(self, prepared, nsparams, rconf, joint_seed):
        """Decompose scalar requests or mixed branches sharing elbow geometry."""
        shoulder, wrist, elbow_angle = prepared
        if isinstance(rconf, (int, np.integer)):
            arm_config, _, wrist_config = self._configuration(rconf)
        else:
            rconf = np.asarray(rconf)
            arm_config = np.where(rconf & 1, -1, 1)
            wrist_config = np.where(rconf & 4, -1, 1)
        psi = np.asarray(nsparams, dtype=float)
        sine, cosine = np.sin(psi)[..., None, None], np.cos(psi)[..., None, None]
        r03 = shoulder[0] * sine + shoulder[1] * cosine + shoulder[2]
        r47 = wrist[0] * sine + wrist[1] * cosine + wrist[2]
        joints = np.empty(psi.shape + (7,))
        joints[..., 0] = np.arctan2(
            r03[..., 1, 1] * arm_config, r03[..., 0, 1] * arm_config
        )
        joints[..., 1] = (
            np.arctan2(np.hypot(r03[..., 0, 1], r03[..., 1, 1]), r03[..., 2, 1])
            * arm_config
        )
        joints[..., 2] = np.arctan2(
            -r03[..., 2, 2] * arm_config, -r03[..., 2, 0] * arm_config
        )
        joints[..., 3] = elbow_angle
        joints[..., 4] = np.arctan2(
            r47[..., 1, 2] * wrist_config, r47[..., 0, 2] * wrist_config
        )
        joints[..., 5] = (
            np.arctan2(np.hypot(r47[..., 0, 2], r47[..., 1, 2]), r47[..., 2, 2])
            * wrist_config
        )
        joints[..., 6] = np.arctan2(
            r47[..., 2, 1] * wrist_config, -r47[..., 2, 0] * wrist_config
        )

        # At singularities both outer joints must be solved together: fixing
        # the first to the seed can discard feasible solutions at joint limits.
        for first, middle, last, rotation in ((0, 1, 2, r03), (4, 5, 6, r47)):
            singular = np.abs(np.sin(joints[..., middle])) < 1e-6
            if np.any(singular):
                d, alpha, a, _ = self.dh_params[first]
                r1 = self._dh_transform(d, alpha, a, 0.0)[:3, :3]
                d, alpha, a, _ = self.dh_params[middle]
                r2 = self._dh_transform(d, alpha, a, joints[..., middle])[..., :3, :3]
                residual = np.swapaxes(r1 @ r2, -1, -2) @ rotation
                sign = np.where(np.cos(joints[..., middle]) >= 0.0, 1.0, -1.0)
                coupled = sign * np.arctan2(residual[..., 1, 0], residual[..., 0, 0])
                coupled -= self.dh_params[first, 3] + sign * self.dh_params[last, 3]
                outer1, outer2 = self._solve_coupled_joints(
                    coupled, sign, first, last, joint_seed
                )
                joints[..., first] = np.where(
                    singular, outer1 + self.dh_params[first, 3], joints[..., first]
                )
                joints[..., last] = np.where(
                    singular, outer2 + self.dh_params[last, 3], joints[..., last]
                )
        return joints - self.dh_params[:, 3]

    def _solve_coupled_joints(self, coupled, sign, first, last, seed):
        """Minimize weighted seed distance for x + sign*y = coupled (mod 2pi).

        The distance to the joint-limit rectangle is convex in the coupled
        angle. Only the two lattice points around the box-projected seed sum
        can minimize it, independent of how many revolutions the limits span.
        """
        physical_limits = self.qpos_limits_np * self.rotation_directions_np[:, None]
        lower, upper = physical_limits.min(axis=1), physical_limits.max(axis=1)
        ylo = np.minimum(sign * lower[last], sign * upper[last])
        yhi = np.maximum(sign * lower[last], sign * upper[last])
        period = 2.0 * np.pi
        kmin = np.ceil((lower[first] + ylo - coupled - 1e-12) / period)
        kmax = np.floor((upper[first] + yhi - coupled + 1e-12) / period)
        nearest_sum = np.clip(seed[first], lower[first], upper[first]) + np.clip(
            sign * seed[last], ylo, yhi
        )
        k = np.floor((nearest_sum - coupled) / period)
        sums = coupled[..., None] + period * np.clip(
            np.stack((k, k + 1), axis=-1), kmin[..., None], kmax[..., None]
        )
        lo = np.maximum(lower[first], sums - yhi[..., None])
        hi = np.minimum(upper[first], sums - ylo[..., None])
        weights = (
            np.ones(2)
            if self.ik_nearst_weight is None
            else self.ik_nearst_weight[[first, last]].copy()
        )
        scale = np.max(weights)
        weights = (weights / scale) ** 2 if scale > 0 else np.ones(2)
        x = (
            weights[0] * seed[first]
            + weights[1] * (sums - sign[..., None] * seed[last])
        ) / weights.sum()
        x = np.clip(x, lo, hi)
        y = sign[..., None] * (sums - x)
        costs = weights[0] * (x - seed[first]) ** 2 + weights[1] * (y - seed[last]) ** 2
        index = np.argmin(costs, axis=-1)[..., None]
        x = np.take_along_axis(x, index, axis=-1)[..., 0]
        y = np.take_along_axis(y, index, axis=-1)[..., 0]
        return np.where(kmin <= kmax, x, np.nan), np.where(kmin <= kmax, y, np.nan)

    def _map_to_limits(self, q, joint_seed, active_qmask=None):
        """Map (..., dof) candidates independently in O(candidates * dof).

        For non-negative weights, each periodic joint can minimize its own
        seed distance; a Cartesian product of periodic values is unnecessary.
        """
        period = 2.0 * np.pi
        tolerance = 1e-6
        lower, upper = self.qpos_limits_np.T
        minimum = np.ceil((lower - q - tolerance) / period)
        maximum = np.floor((upper - q + tolerance) / period)
        nearest = np.floor((joint_seed - q) / period + 0.5)
        turns = np.clip(nearest, minimum, maximum)
        mapped = np.clip(q + period * turns, lower, upper)
        active = (
            np.ones(self.dof, dtype=bool)
            if active_qmask is None
            else active_qmask.astype(bool)
        )
        valid = np.all((minimum <= maximum) | ~active, axis=-1)
        return np.where(active, mapped, q), valid

    def _get_arm_angle(self, qpos: np.ndarray) -> float:
        """Measure the elbow-plane rotation about the shoulder-wrist axis."""
        qpos = np.asarray(qpos, dtype=float) * self.rotation_directions_np
        poses = self._fk_chain(qpos, world=False)
        target = poses[-1]
        elbow_angle = (qpos[3] + self.dh_params[3, 3] + np.pi) % (2 * np.pi) - np.pi
        elbow = -1 if elbow_angle < 0.0 else 1
        reference_rotation, _ = self._reference_geometry(target, elbow)
        if reference_rotation is None:
            raise ValueError("arm angle is undefined at this configuration")
        actual_rotation = poses[2][:3, :3]
        wrist = target[:3, 3] - target[:3, 2] * self.dh_params[-1, 0]
        axis = wrist - np.array([0.0, 0.0, self.link_lengths[0]])
        axis /= np.linalg.norm(axis)
        projections = reference_rotation - np.outer(axis, axis @ reference_rotation)
        norms = np.linalg.norm(projections, axis=0)
        column = int(np.argmax(norms))
        reference = projections[:, column] / norms[column]
        actual = actual_rotation[:, column]
        actual = actual - axis * axis.dot(actual)
        actual /= np.linalg.norm(actual)
        # Scalar triple product avoids np.cross's broadcasting machinery.
        signed_sine = (
            axis[0] * (reference[1] * actual[2] - reference[2] * actual[1])
            + axis[1] * (reference[2] * actual[0] - reference[0] * actual[2])
            + axis[2] * (reference[0] * actual[1] - reference[1] * actual[0])
        )
        return float(np.arctan2(signed_sine, reference.dot(actual)))

    def _get_fk(self, qpos):
        joints = np.asarray(qpos, dtype=float) * self.rotation_directions_np
        pose = np.broadcast_to(self.cfg.T_b_ob, joints.shape[:-1] + (4, 4))
        for transform in self._iter_joint_transforms(joints):
            pose = pose @ transform
        return pose @ self.flange_to_ee

    def _local_target(self, target):
        return self.T_b_ob_inv_np @ target @ self.tcp_inv_np @ self.T_e_oe_inv_np

    def _get_each_ik(
        self, target_pose, nsparam, config, joint_seed=None, prepared=None
    ):
        seed = (
            np.zeros(7) if joint_seed is None else np.asarray(joint_seed, dtype=float)
        )
        if prepared is None:
            prepared = self._prepare_ik(self._local_target(target_pose), config[1])
        if prepared is None:
            return False, None
        rconf = (
            int(config[0] < 0) | (int(config[1] < 0) << 1) | (int(config[2] < 0) << 2)
        )
        joints = (
            self._candidate_joint_angles(
                prepared, nsparam, rconf, seed * self.rotation_directions_np
            )
            * self.rotation_directions_np
        )
        joints, valid = self._map_to_limits(joints, seed)
        return (True, joints) if valid else (False, None)

    def _valid_poses(self, joints, target):
        actual = self._get_fk(joints)
        position = np.linalg.norm(actual[..., :3, 3] - target[:3, 3], axis=-1)
        rotation = np.linalg.norm(actual[..., :3, :3] - target[:3, :3], axis=(-2, -1))
        return (position <= 1e-4) & (rotation <= np.sqrt(8.0) * np.sin(5e-5))

    def _configuration_candidates(self, local, angles, branches, seed):
        """Decompose mixed branch/angle requests in their original input order."""
        output = np.zeros((len(angles), 7))
        success = np.zeros(len(angles), dtype=bool)
        for elbow in (1, -1):
            rows = np.flatnonzero((branches & 2) == (0 if elbow > 0 else 2))
            if not len(rows):
                continue
            geometry = self._prepare_ik(local, elbow)
            if geometry is None:
                continue
            candidates = (
                self._candidate_joint_angles(
                    geometry,
                    angles[rows],
                    branches[rows],
                    seed * self.rotation_directions_np,
                )
                * self.rotation_directions_np
            )
            candidates, valid = self._map_to_limits(candidates, seed)
            output[rows[valid]] = candidates[valid]
            success[rows] = valid
        return success, output

    def _joint_limit_arm_angles(self, prepared):
        """Boundary events and open-cell midpoints for one elbow geometry.

        Each decomposed joint meets a limit where a*sin(psi)+b*cos(psi)+c
        vanishes. Both rays of atan2 are retained: extra cuts are harmless and
        keep the event set independent of the shoulder and wrist branch signs.
        Singular events are checked separately from their neighboring cells.
        """
        shoulder, wrist, _ = prepared
        physical = self.qpos_limits_np * self.rotation_directions_np[:, None]
        lower = physical.min(axis=1) + self.dh_params[:, 3]
        upper = physical.max(axis=1) + self.dh_params[:, 3]
        key = (lower.tobytes(), upper.tobytes())
        cached = getattr(self, "_ik_boundary_cache", None)
        entries = [] if cached is None or cached[0] != key else cached[1]
        for geometry, angles in entries:
            if geometry is prepared:
                return angles
        shoulder, wrist = np.asarray(shoulder), np.asarray(wrist)
        equations = []
        for joint, y, x in (
            (0, shoulder[:, 1, 1], shoulder[:, 0, 1]),
            (2, -shoulder[:, 2, 2], -shoulder[:, 2, 0]),
            (4, wrist[:, 1, 2], wrist[:, 0, 2]),
            (6, wrist[:, 2, 1], -wrist[:, 2, 0]),
        ):
            if upper[joint] - lower[joint] < 2 * np.pi:
                for limit in (lower[joint], upper[joint]):
                    equations.append(y * np.cos(limit) - x * np.sin(limit))
        for first, middle, last, rotation, z in (
            (0, 1, 2, shoulder, shoulder[:, 2, 1]),
            (4, 5, 6, wrist, wrist[:, 2, 2]),
        ):
            values = [-1.0, 1.0]
            if upper[middle] - lower[middle] < 2 * np.pi:
                values.extend(np.cos([lower[middle], upper[middle]]))
            for value in values:
                equation = z.copy()
                equation[2] -= value
                equations.append(equation)
            # A sphere can stay singular over the whole circle, e.g. at full
            # extension. Its coupled outer-joint sum then determines feasibility.
            if np.hypot(*z[:2]) < 1e-13 and abs(abs(z[2]) - 1) < 1e-13:
                angle = np.arccos(np.clip(z[2], -1.0, 1.0))
                d, alpha, a, _ = self.dh_params[first]
                r1 = self._dh_transform(d, alpha, a, 0.0)[:3, :3]
                d, alpha, a, _ = self.dh_params[middle]
                r2 = self._dh_transform(d, alpha, a, angle)[:3, :3]
                residual = (r1 @ r2).T @ rotation
                sign = 1.0 if z[2] >= 0 else -1.0
                limits = (
                    lower[first] + min(sign * lower[last], sign * upper[last]),
                    upper[first] + max(sign * lower[last], sign * upper[last]),
                )
                for limit in limits:
                    equations.append(
                        residual[:, 1, 0] * np.cos(sign * limit)
                        - residual[:, 0, 0] * np.sin(sign * limit)
                    )
        roots = self._harmonic_arm_angle_roots(equations)
        cuts = np.unique(np.r_[-np.pi, roots, np.pi])
        angles = np.unique(np.r_[cuts, cuts[:-1] + np.diff(cuts) / 2])
        angles.setflags(write=False)
        self._ik_boundary_cache = (key, (entries + [(prepared, angles)])[-2:])
        return angles

    @staticmethod
    def _harmonic_arm_angle_roots(equations):
        """Solve harmonic zeros via a stable tan(psi/2) quadratic.

        Computing phase +/- acos instead can lose the small arm-angle root to
        cancellation, which matters when an outer joint has a fixed limit.
        """
        a, b, c = np.asarray(equations).T
        amplitude = np.hypot(a, b)
        valid = (amplitude > 0) & (np.abs(c) <= amplitude * (1 + 1e-12))
        a, b, c, amplitude = (v[valid] for v in (a, b, c, amplitude))
        discriminant = np.maximum((amplitude - np.abs(c)) * (amplitude + np.abs(c)), 0)
        q = -a - np.copysign(np.sqrt(discriminant), a)
        first = np.divide(q, c - b, out=np.full_like(q, np.inf), where=c != b)
        second = np.divide(b + c, q, out=first.copy(), where=q != 0)
        return 2 * np.arctan(np.r_[first, second])

    def _joint_limit_candidates(self, local_target, joint_seed):
        """Evaluate limit-boundary events for all eight discrete branches."""
        angles, branches = [], []
        for elbow in (1, -1):
            prepared = self._prepare_ik(local_target, elbow)
            if prepared is None:
                continue
            psi = self._joint_limit_arm_angles(prepared)
            for branch in range(8):
                if (-1 if branch & 2 else 1) == elbow:
                    angles.extend(psi)
                    branches.extend([branch] * len(psi))
        return self._configuration_candidates(
            local_target,
            np.asarray(angles),
            np.asarray(branches, dtype=int),
            joint_seed,
        )

    def _joint_limit_anchor(self, target, seed):
        """Try box anchors when the arm-angle parameterization is degenerate."""
        lower, upper = self.qpos_limits_np.T
        for joints in (0.5 * lower + 0.5 * upper, np.clip(seed, lower, upper)):
            joints = joints.astype(np.float32)
            if self._valid_poses(joints, target):
                return joints.copy()
        return None

    def get_ik(
        self,
        target_xpos,
        qpos_seed,
        return_all_solutions=False,
        search_mode="continuous",
        **kwargs,
    ):
        self._refresh_weights()
        count = target_xpos.shape[0]
        seeds = (
            np.zeros((count, 7))
            if qpos_seed is None
            else np.asarray(qpos_seed.detach().cpu().numpy(), dtype=np.float64)
        )
        targets = target_xpos.detach().cpu().numpy()
        slots = 8 * self.cfg.num_samples if return_all_solutions else 1
        output = np.zeros((count, slots, 7), dtype=np.float32)
        success = np.zeros(count, dtype=bool)
        for row, (target, seed) in enumerate(zip(targets, seeds)):
            mapped, valid = self._map_to_limits(seed, seed)
            mapped = mapped.astype(np.float32)
            if (
                not return_all_solutions
                and valid
                and np.allclose(self._get_fk(mapped), target, atol=1e-7, rtol=0)
            ):
                output[row, 0], success[row] = mapped, True
                continue
            local_target = self._local_target(target)
            geometry = {1: self._prepare_ik(local_target, 1)}
            if geometry[1] is None:
                if search_mode != "global" and not return_all_solutions:
                    anchor = self._joint_limit_anchor(target, seed)
                    if anchor is not None:
                        output[row, 0], success[row] = anchor, True
                continue

            def prepare(branch):
                elbow = -1 if branch & 2 else 1
                if elbow not in geometry:
                    geometry[elbow] = self._prepare_ik(local_target, elbow)
                return geometry[elbow]

            if not return_all_solutions and search_mode == "continuous":
                try:
                    psi = self._get_arm_angle(seed)
                    angles = (
                        seed * self.rotation_directions_np
                        + self.dh_params[:, 3]
                        + np.pi
                    ) % (2 * np.pi) - np.pi
                    branch = (
                        int(angles[1] < 0)
                        | (int(angles[3] < 0) << 1)
                        | (int(angles[5] < 0) << 2)
                    )
                    ok, joints = self._get_each_ik(
                        target, psi, self.configs[branch], seed, prepare(branch)
                    )
                    if ok and self._valid_poses(joints.astype(np.float32), target):
                        output[row, 0], success[row] = joints, True
                        continue
                    offsets = np.array([0, -1, 1, -2, 2, -3, 3, -4, 4]) * np.deg2rad(5)
                    order = sorted(
                        range(8),
                        key=lambda other: (
                            (other ^ branch) & 2 != 0,
                            bin(other ^ branch).count("1"),
                        ),
                    )
                    for other in order:
                        psis = psi + (offsets[1:] if other == branch else offsets)
                        candidates = (
                            self._candidate_joint_angles(
                                prepare(other),
                                psis,
                                other,
                                seed * self.rotation_directions_np,
                            )
                            * self.rotation_directions_np
                        )
                        candidates, valid = self._map_to_limits(candidates, seed)
                        candidates = candidates[valid].astype(np.float32)
                        valid = self._valid_poses(candidates, target)
                        if np.any(valid):
                            output[row, 0], success[row] = candidates[valid][0], True
                            break
                    if success[row]:
                        continue
                except (ValueError, FloatingPointError):
                    pass
            searches = ("interval",) if search_mode == "interval" else ("global",)
            if search_mode == "continuous" and not return_all_solutions:
                searches += ("interval",)
            for search in searches:
                if search == "interval":
                    valid, joints = self._joint_limit_candidates(local_target, seed)
                else:
                    branches = np.repeat(np.arange(8), len(self.elbow_angles))
                    valid, joints = self._configuration_candidates(
                        local_target, np.tile(self.elbow_angles, 8), branches, seed
                    )
                candidates = joints[valid].astype(np.float32)
                if self.cfg.sort_ik:
                    distances = np.linalg.norm(
                        (candidates - seed) * self._distance_weights, axis=-1
                    )
                    candidates = candidates[np.argsort(distances, kind="stable")]
                if return_all_solutions:
                    candidates = candidates[self._valid_poses(candidates, target)]
                    output[row, : len(candidates)] = candidates
                    success[row] = len(candidates) > 0
                else:
                    for joints in candidates:
                        if self._valid_poses(joints, target):
                            output[row, 0], success[row] = joints, True
                            break
                if success[row]:
                    break
            if (
                not success[row]
                and search_mode != "global"
                and not return_all_solutions
            ):
                anchor = self._joint_limit_anchor(target, seed)
                if anchor is not None:
                    output[row, 0], success[row] = anchor, True
        return _as_tensor(success, device=self.device), _as_tensor(
            output, device=self.device
        )


class _WarpSRSSolverImpl(_BaseSRSSolverImpl):
    """Warp implementation for CUDA or CPU devices."""

    def _parse_params(self):
        super()._parse_params()

        # Convert numpy transformation matrices to Warp mat44 format for CUDA computation.
        self.tcp_inv_wp = wp.mat44(*self.tcp_inv_np.flatten())
        self.T_b_ob_inv_wp = wp.mat44(*self.T_b_ob_inv_np.flatten())
        self.T_e_oe_inv_wp = wp.mat44(*self.T_e_oe_inv_np.flatten())

        # Convert DH parameters, joint limits, link lengths, and rotation directions to Warp arrays.
        self.dh_params_wp = wp.array(
            self.dh_params_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self.link_lengths_wp = wp.array(
            self.link_lengths_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self.rotation_directions_wp = wp.array(
            self.rotation_directions_np.flatten(),
            dtype=float,
            device=standardize_device_string(self.device),
        )

        # Branch order is part of deterministic solution tie breaking.
        self.configs = [
            wp.vec3(
                1 if i & 1 == 0 else -1,
                1 if i & 2 == 0 else -1,
                1 if i & 4 == 0 else -1,
            )
            for i in range(8)
        ]
        self.configs_wp = wp.array(
            self.configs, dtype=wp.vec3, device=standardize_device_string(self.device)
        )

        self.elbow_angles = np.linspace(
            -wp.pi, wp.pi, self.cfg.num_samples, endpoint=False
        ).tolist()
        self.elbow_angles_wp = wp.array(
            self.elbow_angles,
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self._weight_values = None
        self._refresh_weights()
        self._workspace_key = None
        self._workspace = {}
        self._workspace_views = {}
        self._validation_error = wp.zeros(
            1, dtype=int, device=standardize_device_string(self.device)
        )
        self._validation_status = _torch_view(self._validation_error)

    def _refresh_weights(self):
        """Refresh device constants after a live configuration weight edit."""
        weights = np.asarray(self.cfg.ik_nearest_weight, dtype=float)
        if np.array_equal(weights, self._weight_values):
            return
        super()._refresh_weights()
        self._weight_values = weights.copy()
        # Singular shoulder/wrist pairs need only their local weight ratio.
        # Normalize each pair independently so other joints cannot erase it.
        singular_weights = np.zeros(7)
        for pair in ([0, 2], [4, 6]):
            weights = self.ik_nearst_weight[pair]
            scale = np.max(weights)
            singular_weights[pair] = weights / scale if scale > 0 else weights
        self.nearest_weights_wp = wp.array(
            singular_weights,
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self.distance_weights_wp = wp.array(
            self._distance_weights,
            dtype=float,
            device=standardize_device_string(self.device),
        )

    def _ensure_workspace(self, num_targets: int, num_combinations: int) -> dict:
        """Reuse large device buffers across control-loop calls of equal shape."""
        key = (num_targets, num_combinations)
        if key == self._workspace_key:
            return self._workspace
        device = standardize_device_string(self.device)
        self._workspace = {
            "xpos": wp.empty(num_targets, dtype=wp.mat44, device=device),
            "geometry": wp.empty(num_targets * 2, dtype=IKGeometry, device=device),
            "qpos": wp.empty(num_combinations * 7, dtype=float, device=device),
            "success": wp.empty(num_combinations, dtype=int, device=device),
        }
        self._workspace_views = {
            name: _torch_view(self._workspace[name]) for name in ("qpos", "success")
        }
        self._workspace_key = key
        return self._workspace

    def _solution_distances(self, seed, num_per_target):
        """Reuse one scalar score per candidate; Torch performs the stable reduction."""
        success_wp = self._workspace["success"]
        distances = self._workspace.get("distances")
        if distances is None:
            distances = wp.empty(
                success_wp.shape[0],
                dtype=float,
                device=standardize_device_string(self.device),
            )
            self._workspace["distances"] = distances
            self._workspace_views["distances"] = _torch_view(distances)
        wp.launch(
            ik_distance_kernel,
            dim=success_wp.shape[0],
            inputs=[
                self._workspace["qpos"],
                success_wp,
                wp.from_torch(seed.contiguous().flatten()),
                self.distance_weights_wp,
                num_per_target,
            ],
            outputs=[distances],
            device=standardize_device_string(self.device),
        )
        return self._workspace_views["distances"].view(-1, num_per_target)

    def _select_solutions(self, seed, count, candidates, return_all):
        """Rank valid candidates and return independent, zero-masked outputs."""
        joints = self._workspace_views["qpos"].view(count, candidates, 7)
        valid = self._workspace_views["success"].view(count, candidates).bool()
        success = valid.any(dim=1)
        if self.cfg.sort_ik:
            distances = self._solution_distances(seed, candidates)
            if return_all:
                order = distances.argsort(dim=1, stable=True)
                joints = torch.gather(joints, 1, order[:, :, None].expand(-1, -1, 7))
                valid = torch.gather(valid, 1, order)
            else:
                index = distances.argmin(dim=1)
        elif not return_all:
            index = valid.to(torch.int32).argmax(dim=1)
        if return_all:
            return success, joints.masked_fill(~valid[:, :, None], 0.0)
        rows = torch.arange(count, device=self.device)
        nearest = joints[rows, index].masked_fill(~success[:, None], 0.0)
        return success, nearest[:, None, :]

    def _validate_inputs(self, targets, seed):
        """Report the same input errors with one device-to-host status read."""
        self._validation_error.zero_()
        wp.launch(
            validate_ik_inputs_kernel,
            dim=targets.shape[0],
            inputs=[
                wp.from_torch(targets.contiguous(), dtype=wp.mat44),
                (
                    wp.from_torch(seed.contiguous().flatten())
                    if seed is not None
                    else None
                ),
                int(seed is not None),
            ],
            outputs=[self._validation_error],
            device=standardize_device_string(self.device),
        )
        code = int(self._validation_status.item())
        if code == 3:
            raise ValueError("target_xpos must contain only finite values")
        if code == 2:
            raise ValueError("target_xpos must contain rigid homogeneous transforms")
        if code == 1:
            raise ValueError("qpos_seed must contain only finite values")

    def get_ik(self, target_xpos, qpos_seed, return_all_solutions=False, **kwargs):
        """Evaluate the global grid; inputs are validated by SRSSolver.get_ik."""
        self._refresh_weights()
        targets = wp.from_torch(target_xpos, dtype=wp.mat44)
        count = target_xpos.shape[0]
        candidates = len(self.configs) * len(self.elbow_angles)
        workspace = self._ensure_workspace(count, count * candidates)
        device = standardize_device_string(self.device)
        wp.launch(
            prepare_ik_geometry_kernel,
            dim=count * 2,
            inputs=[
                targets,
                self.T_b_ob_inv_wp,
                self.T_e_oe_inv_wp,
                self.tcp_inv_wp,
                self.dh_params_wp,
                self.link_lengths_wp,
            ],
            outputs=[workspace["xpos"], workspace["geometry"]],
            device=device,
        )
        if qpos_seed is None:
            qpos_seed = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
        wp.launch(
            compute_prepared_ik_kernel,
            dim=count * candidates,
            inputs=[
                workspace["xpos"],
                workspace["geometry"],
                self.elbow_angles_wp,
                self.qpos_limits_wp,
                self.configs_wp,
                self.dh_params_wp,
                self.rotation_directions_wp,
                wp.from_torch(qpos_seed.flatten()),
                self.nearest_weights_wp,
            ],
            outputs=[workspace["success"], workspace["qpos"]],
            device=device,
        )
        return self._select_solutions(
            qpos_seed,
            count,
            candidates,
            return_all_solutions,
        )


class SRSSolver:
    """Analytical SRS FK/IK with NumPy or Warp batch execution."""

    def __init__(
        self,
        cfg: SRSSolverCfg,
        num_envs: int,
        device: str | torch.device | None,
        use_warp: bool | None = None,
        **kwargs,
    ):
        """Validate configuration and initialize the selected batch backend."""

        self.cfg = cfg
        self.cfg.validate()
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = _resolve_device(device)
        self.use_warp = self.device.type == "cuda" if use_warp is None else use_warp
        if self.device.type == "cuda" and not self.use_warp:
            raise ValueError("CUDA execution requires the Warp backend")
        if self.use_warp:
            _ensure_writable_warp_cache()
            wp.init()

        self.urdf_path = getattr(cfg, "urdf_path", None)
        self.joint_names = getattr(cfg, "joint_names", None)
        self.end_link_name = getattr(cfg, "end_link_name", None)
        self.root_link_name = getattr(cfg, "root_link_name", None)

        # Degrees of freedom
        self.dofs = 7

        self.dof = len(self.joint_names) if self.joint_names else self.dofs

        self.ik_nearest_weight = torch.tensor(
            cfg.ik_nearest_weight, dtype=torch.float32, device=self.device
        )
        self.tcp_xpos = np.eye(4)
        self.root_base_xpos = np.eye(4)

        # Initialize joint position limits from cfg
        qpos_limits = np.asarray(cfg.qpos_limits)  # shape (7, 2)
        self.lower_qpos_limits = torch.tensor(
            qpos_limits[:, 0], dtype=torch.float32, device=self.device
        )
        self.upper_qpos_limits = torch.tensor(
            qpos_limits[:, 1], dtype=torch.float32, device=self.device
        )

        with _torch_stream(self.device):
            # Initialize implementation based on device
            if self.use_warp:
                self.impl = _WarpSRSSolverImpl(cfg, self.device)
            else:
                self.impl = _CPUSRSSolverImpl(cfg, self.device)

            self._update_impl_qpos_limits()

        # Explicit branch/arm-angle queries are inexpensive scalar geometry.
        # Reuse the CPU implementation even when the batch solver is on CUDA.
        self._geometry_impl = (
            self.impl
            if not self.use_warp
            else _CPUSRSSolverImpl(cfg, torch.device("cpu"))
        )
        self._geometry_impl.qpos_limits_np = self.impl.qpos_limits_np.copy()

    def get_dof(self) -> int:
        return self.dofs

    def get_position_limits(self) -> dict[str, np.ndarray]:
        return {
            "lower": self.lower_qpos_limits.detach().cpu().numpy().copy(),
            "upper": self.upper_qpos_limits.detach().cpu().numpy().copy(),
        }

    def set_position_limits(self, lower, upper) -> bool:
        lower_array = np.asarray(lower, dtype=np.float32)
        upper_array = np.asarray(upper, dtype=np.float32)
        if lower_array.shape != (7,) or upper_array.shape != (7,):
            return False
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            return False
        if np.any(lower_array > upper_array):
            return False
        self.update_with_robot_limit(np.column_stack((lower_array, upper_array)))
        return True

    def get_tcp(self) -> np.ndarray:
        return np.asarray(self.cfg.tcp).copy()

    def set_tcp(self, transform: np.ndarray | torch.Tensor) -> None:
        """Set the flange-to-tool transform and refresh cached device constants."""
        tcp = np.asarray(_as_tensor(transform).detach().cpu(), dtype=np.float64)
        _validate_pose(tcp)
        self.cfg.tcp = tcp.copy()
        with _torch_stream(self.device):
            if self.use_warp:
                self.impl = _WarpSRSSolverImpl(self.cfg, self.device)
                self._geometry_impl = _CPUSRSSolverImpl(self.cfg, torch.device("cpu"))
            else:
                self.impl = _CPUSRSSolverImpl(self.cfg, self.device)
                self._geometry_impl = self.impl
            self._update_impl_qpos_limits()

    def get_arm_angle(self, qpos: np.ndarray | torch.Tensor) -> float:
        """Measure elbow-plane rotation around the shoulder-to-wrist axis."""
        joints = np.asarray(_as_tensor(qpos).detach().cpu(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("qpos must be a finite array of shape (7,)")
        return self._geometry_impl._get_arm_angle(joints)

    def get_configuration(self, qpos: np.ndarray | torch.Tensor) -> SRSConfiguration:
        """Return the current S/E/W branch and continuous arm angle."""
        joints = np.asarray(_as_tensor(qpos).detach().cpu(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("qpos must be a finite array of shape (7,)")

        joints = (
            joints * self.cfg.rotation_directions + self.cfg.dh_params[:, 3] + np.pi
        ) % (2 * np.pi) - np.pi

        def sign(value):
            return -1 if value < 0.0 else 1

        return SRSConfiguration(
            sign(joints[1]),
            sign(joints[3]),
            sign(joints[5]),
            self.get_arm_angle(qpos),
        )

    def solve_configuration(
        self,
        target_xpos: np.ndarray | torch.Tensor,
        configuration: SRSConfiguration,
        joint_seed: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[bool, torch.Tensor | None]:
        """Solve one explicitly selected branch at an exact arm angle."""
        target = np.asarray(_as_tensor(target_xpos).detach().cpu(), dtype=np.float64)
        _validate_pose(target, "target_xpos")
        if not isinstance(configuration, SRSConfiguration):
            raise TypeError("configuration must be an SRSConfiguration")
        seed = (
            np.zeros(7)
            if joint_seed is None
            else np.asarray(_as_tensor(joint_seed).detach().cpu(), dtype=float)
        )
        if seed.shape != (7,) or not np.all(np.isfinite(seed)):
            raise ValueError("joint_seed must be a finite array of shape (7,)")
        branch = np.array(
            [configuration.shoulder, configuration.elbow, configuration.wrist],
            dtype=np.float64,
        )
        self._geometry_impl._refresh_weights()
        success, joints = self._geometry_impl._get_each_ik(
            target, configuration.redundancy, branch, seed
        )
        if not success:
            return False, None
        joints = joints.astype(np.float32)
        actual = self._geometry_impl._get_fk(joints)
        if not np.allclose(actual, target, atol=1e-4, rtol=0.0):
            return False, None
        return True, _as_tensor(joints, dtype=torch.float32, device=self.device)

    def _update_impl_qpos_limits(self):
        with _torch_stream(self.device):
            qpos_limits = torch.vstack(
                [self.lower_qpos_limits, self.upper_qpos_limits]
            ).T
            self.impl.qpos_limits_np = qpos_limits.cpu().numpy()
            if self.use_warp:
                self.impl.qpos_limits_wp = wp.array(
                    self.impl.qpos_limits_np,
                    dtype=wp.vec2,
                    device=standardize_device_string(self.device),
                )
            if hasattr(self, "_geometry_impl"):
                self._geometry_impl.qpos_limits_np = self.impl.qpos_limits_np.copy()

    def solve_configurations(self, target_xpos, configurations, joint_seed=None):
        """Solve exact configurations for one target and shared seed on CPU geometry.

        Returns device tensors: boolean (N,) success and float32 (N, 7) joints,
        preserving input order and zeroing invalid rows. Geometry is prepared at
        most once per elbow branch; arm angles and FK validation are batched.
        """
        target = np.asarray(_as_tensor(target_xpos).detach().cpu(), dtype=float)
        _validate_pose(target, "target_xpos")
        seed = (
            np.zeros(7)
            if joint_seed is None
            else np.asarray(_as_tensor(joint_seed).detach().cpu(), dtype=float)
        )
        if seed.shape != (7,) or not np.all(np.isfinite(seed)):
            raise ValueError("joint_seed must be a finite array of shape (7,)")
        configurations, branches, angles = self._prepare_configuration_requests(
            configurations
        )
        self._geometry_impl._refresh_weights()
        if not configurations:
            output = np.zeros((0, 7))
            success = np.zeros(0, dtype=bool)
        else:
            impl = self._geometry_impl
            local = impl._local_target(target)
            success, output = impl._configuration_candidates(
                local, angles, branches, seed
            )
            output = output.astype(np.float32)
            rows = np.flatnonzero(success)
            if len(rows):
                actual = impl._get_fk(output[rows])
                success[rows] = np.all(np.abs(actual - target) <= 1e-4, axis=(-2, -1))
            output[~success] = 0
        return (
            _as_tensor(success, device=self.device),
            _as_tensor(output, dtype=torch.float32, device=self.device),
        )

    def _prepare_configuration_requests(self, configurations):
        """Reuse one batch of unchanged frozen configuration objects."""
        configurations = tuple(configurations)
        cached = getattr(self, "_configuration_request_cache", None)
        if cached is not None and (
            configurations is cached[0]
            or len(configurations) == len(cached[0])
            and all(
                current is previous
                for current, previous in zip(configurations, cached[0])
            )
        ):
            return cached
        if any(not isinstance(c, SRSConfiguration) for c in configurations):
            raise TypeError("configurations must contain SRSConfiguration values")
        branches = np.fromiter(
            (
                (c.shoulder < 0) | ((c.elbow < 0) << 1) | ((c.wrist < 0) << 2)
                for c in configurations
            ),
            dtype=int,
            count=len(configurations),
        )
        angles = np.fromiter(
            (c.redundancy for c in configurations),
            dtype=float,
            count=len(configurations),
        )
        branches.setflags(write=False)
        angles.setflags(write=False)
        cached = configurations, branches, angles
        self._configuration_request_cache = cached
        return cached

    def update_with_robot_limit(self, robot_qpos_limits):
        limits = np.asarray(robot_qpos_limits, dtype=np.float32)
        if limits.shape != (7, 2):
            raise ValueError("robot_qpos_limits must have shape (7, 2)")
        if not np.all(np.isfinite(limits)):
            raise ValueError("robot_qpos_limits must be finite")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("each lower joint limit must not exceed its upper limit")
        self.cfg.qpos_limits = limits.copy()
        self.lower_qpos_limits = torch.tensor(
            limits[:, 0], dtype=torch.float32, device=self.device
        )
        self.upper_qpos_limits = torch.tensor(
            limits[:, 1], dtype=torch.float32, device=self.device
        )
        self._update_impl_qpos_limits()

    def get_fk(self, qpos: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute FK for input shape ``(7,)`` or ``(N, 7)``.

        Warp execution launches one thread per robot.  The result always
        has shape ``(N, 4, 4)`` on the solver device.
        """
        qpos_tensor = _as_tensor(qpos, dtype=torch.float32, device=self.device)
        if qpos_tensor.ndim == 1:
            qpos_tensor = qpos_tensor.unsqueeze(0)
        if qpos_tensor.ndim != 2 or qpos_tensor.shape[1] != 7:
            raise ValueError("qpos must have shape (7,) or (N, 7)")
        if qpos_tensor.shape[0] == 0:
            return torch.empty((0, 4, 4), dtype=qpos_tensor.dtype, device=self.device)
        if not self.use_warp:
            if not bool(torch.isfinite(qpos_tensor).all()):
                raise ValueError("qpos must contain only finite values")
            poses = self.impl._get_fk(qpos_tensor.detach().cpu().numpy())
            return _as_tensor(poses, dtype=qpos_tensor.dtype, device=self.device)

        with _torch_stream(self.device):
            warp_device = standardize_device_string(self.device)
            joint_angles = wp.from_torch(qpos_tensor.contiguous().flatten())
            poses = torch.empty(
                (qpos_tensor.shape[0], 4, 4),
                dtype=qpos_tensor.dtype,
                device=self.device,
            )
            success = torch.empty(
                qpos_tensor.shape[0], dtype=torch.int32, device=self.device
            )
            wp.launch(
                compute_fk_kernel,
                dim=qpos_tensor.shape[0],
                inputs=[
                    joint_angles,
                    self.impl.dh_params_wp,
                    self.impl.rotation_directions_wp,
                    wp.mat44(*np.asarray(self.cfg.T_b_ob).flatten()),
                    wp.mat44(*np.asarray(self.cfg.T_e_oe).flatten()),
                    wp.mat44(*np.asarray(self.cfg.tcp).flatten()),
                ],
                outputs=[wp.from_torch(poses, dtype=wp.mat44), wp.from_torch(success)],
                device=warp_device,
            )
            if not bool(success.all()):
                raise ValueError("qpos must contain only finite values")
            return poses

    def get_all_fk_mat(self, qpos: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return (N, 8, 4, 4): seven DH link frames followed by the TCP.

        All poses are in world coordinates. Accepts (7,) or (N, 7), including
        empty batches. Each call owns its outputs, so later calls cannot overwrite them.
        """
        joints = _as_tensor(qpos, dtype=torch.float32, device=self.device)
        if joints.ndim == 1:
            joints = joints.unsqueeze(0)
        if (
            joints.ndim != 2
            or joints.shape[1] != 7
            or not bool(torch.isfinite(joints).all())
        ):
            raise ValueError("qpos must be finite and have shape (7,) or (N, 7)")
        count = joints.shape[0]
        if count == 0:
            return torch.empty((0, 8, 4, 4), dtype=joints.dtype, device=self.device)
        if self.use_warp:
            with _torch_stream(self.device):
                output = torch.empty(
                    (count, 8, 4, 4), dtype=joints.dtype, device=self.device
                )
                wp.launch(
                    compute_fk_chain_kernel,
                    dim=count,
                    inputs=[
                        wp.from_torch(joints.contiguous().flatten()),
                        self.impl.dh_params_wp,
                        self.impl.rotation_directions_wp,
                        wp.mat44(*self.cfg.T_b_ob.ravel()),
                        wp.mat44(*(self.cfg.T_e_oe @ self.cfg.tcp).ravel()),
                    ],
                    outputs=[wp.from_torch(output.view(-1, 4, 4), dtype=wp.mat44)],
                    device=standardize_device_string(self.device),
                )
                return output
        physical = joints.detach().cpu().numpy() * self.cfg.rotation_directions
        pose = np.broadcast_to(self.cfg.T_b_ob, (count, 4, 4))
        poses = np.empty((count, 8, 4, 4))
        for index, transform in enumerate(self.impl._iter_joint_transforms(physical)):
            pose = pose @ transform
            poses[:, index] = pose
        poses[:, 7] = pose @ self.cfg.T_e_oe @ self.cfg.tcp
        return _as_tensor(poses, dtype=joints.dtype, device=self.device)

    def get_jacobian(
        self, qpos: np.ndarray | torch.Tensor, step: float = 1e-4
    ) -> torch.Tensor:
        """Return analytical world-frame geometric Jacobians, linear rows first.

        ``step`` remains accepted for compatibility; finite differences are not used.
        """
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be positive and finite")
        joints = _as_tensor(qpos, dtype=torch.float32, device=self.device)
        squeeze = joints.ndim == 1
        if squeeze:
            joints = joints.unsqueeze(0)
        if (
            joints.ndim != 2
            or joints.shape[1] != 7
            or not bool(torch.isfinite(joints).all())
        ):
            raise ValueError("qpos must be finite and have shape (7,) or (N, 7)")
        batch = joints.shape[0]
        if self.use_warp:
            with _torch_stream(self.device):
                output = torch.empty(
                    (batch, 6, 7), dtype=joints.dtype, device=self.device
                )
                if batch:
                    wp.launch(
                        compute_jacobian_kernel,
                        dim=batch,
                        inputs=[
                            wp.from_torch(joints.contiguous().flatten()),
                            self.impl.dh_params_wp,
                            self.impl.rotation_directions_wp,
                            wp.mat44(*np.asarray(self.cfg.T_b_ob).ravel()),
                            wp.mat44(*(self.cfg.T_e_oe @ self.cfg.tcp).ravel()),
                        ],
                        outputs=[wp.from_torch(output.flatten())],
                        device=standardize_device_string(self.device),
                    )
        else:
            q = joints.detach().cpu().numpy() * self.cfg.rotation_directions
            pose = np.broadcast_to(self.cfg.T_b_ob, (batch, 4, 4))
            origins, axes = np.empty((batch, 7, 3)), np.empty((batch, 7, 3))
            for index, transform in enumerate(self.impl._iter_joint_transforms(q)):
                origins[:, index] = pose[:, :3, 3]
                axes[:, index] = pose[:, :3, 2] * self.cfg.rotation_directions[index]
                pose = pose @ transform
            tcp = (pose @ self.cfg.T_e_oe @ self.cfg.tcp)[:, :3, 3]
            linear = np.cross(axes, tcp[:, None] - origins)
            jacobian = np.empty((batch, 6, 7))
            jacobian[:, :3] = linear.transpose(0, 2, 1)
            jacobian[:, 3:] = axes.transpose(0, 2, 1)
            output = _as_tensor(jacobian, dtype=joints.dtype, device=self.device)
        return output[0] if squeeze else output

    def get_null_space_velocity(
        self,
        qpos: np.ndarray | torch.Tensor,
        preferred_velocity: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Project preferred joint velocities into the FK Jacobian null space."""
        jacobian = self.get_jacobian(qpos)
        squeeze = jacobian.ndim == 2
        if squeeze:
            jacobian = jacobian.unsqueeze(0)
        preferred = _as_tensor(
            preferred_velocity, dtype=jacobian.dtype, device=self.device
        )
        if preferred.ndim == 1:
            preferred = preferred.unsqueeze(0)
        if preferred.shape != (jacobian.shape[0], 7):
            raise ValueError("preferred_velocity must have shape (7,) or (N, 7)")
        if not bool(torch.isfinite(preferred).all()):
            raise ValueError("preferred_velocity must contain finite values")
        # Project through orthonormal null-space vectors directly. Forming
        # I - pinv(J) @ J amplifies roundoff when J is nearly singular.
        _, singular, vh = torch.linalg.svd(jacobian, full_matrices=True)
        cutoff = (
            singular[:, :1] * max(jacobian.shape[-2:]) * torch.finfo(jacobian.dtype).eps
        )
        null = torch.cat(
            (singular <= cutoff, torch.ones_like(singular[:, :1], dtype=torch.bool)),
            dim=1,
        )
        # Elementwise reductions preserve float32 inside the caller's autocast scope.
        coefficients = (vh * preferred[:, None, :]).sum(dim=2)
        coefficients = coefficients.masked_fill(~null, 0.0)
        velocity = (vh * coefficients[:, :, None]).sum(dim=1)
        return velocity[0] if squeeze else velocity

    def get_null_space_direction(self, qpos: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return a deterministic unit vector in each Jacobian null space."""
        jacobian = self.get_jacobian(qpos)
        squeeze = jacobian.ndim == 2
        if squeeze:
            jacobian = jacobian.unsqueeze(0)
        direction = torch.linalg.svd(jacobian, full_matrices=True).Vh[:, -1, :]
        pivot = direction.abs().argmax(dim=1)
        pivot_value = direction.gather(1, pivot[:, None]).squeeze(1)
        direction = torch.where((pivot_value < 0.0)[:, None], -direction, direction)
        direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        return direction[0] if squeeze else direction

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor = None,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad). Default is None.
            return_all_solutions: Whether to return all solutions. Default is False.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        search_mode = kwargs.pop(
            "search_mode", "global" if self.use_warp else "continuous"
        )
        if kwargs:
            raise TypeError(f"Unexpected IK arguments: {', '.join(sorted(kwargs))}")
        if search_mode not in ("continuous", "global", "interval"):
            raise ValueError("search_mode must be 'continuous', 'global' or 'interval'")
        if search_mode == "interval" and return_all_solutions:
            raise ValueError("interval search returns one solution per target")
        if self.use_warp and search_mode == "continuous":
            raise ValueError(
                "continuous search requires use_warp=False; choose global for native "
                "Warp or interval for CPU boundary search"
            )
        interval_search = search_mode == "interval"
        input_device = torch.device("cpu") if interval_search else self.device
        input_dtype = torch.float64 if interval_search else torch.float32
        target_xpos = _as_tensor(target_xpos, dtype=input_dtype, device=input_device)
        if target_xpos.ndim == 2:
            target_xpos = target_xpos.unsqueeze(0)
        if target_xpos.ndim != 3 or target_xpos.shape[1:] != (4, 4):
            raise ValueError("target_xpos must have shape (4, 4) or (N, 4, 4)")
        if qpos_seed is not None:
            qpos_seed = _as_tensor(qpos_seed, dtype=input_dtype, device=input_device)
            if qpos_seed.ndim == 1:
                qpos_seed = qpos_seed.unsqueeze(0)
            if qpos_seed.shape != (target_xpos.shape[0], 7):
                raise ValueError("qpos_seed must have shape (7,) or (N, 7)")
            qpos_seed = qpos_seed.contiguous()
        target_xpos = target_xpos.contiguous()
        if target_xpos.shape[0] == 0:
            slots = 8 * self.cfg.num_samples if return_all_solutions else 1
            return torch.empty(0, dtype=torch.bool, device=self.device), torch.empty(
                (0, slots, 7), dtype=torch.float32, device=self.device
            )
        with _torch_stream(self.device):
            impl = self._geometry_impl if interval_search else self.impl
            impl._validate_inputs(target_xpos, qpos_seed)
            if interval_search:
                # Boundary-event search uses scalar float64 CPU geometry, just
                # like exact configuration queries. Native global batching is
                # unchanged; this explicit mode favors narrow-limit reliability.
                success, joints = self._geometry_impl.get_ik(
                    target_xpos, qpos_seed, search_mode=search_mode
                )
                return success.to(self.device), joints.to(self.device)
            return self.impl.get_ik(
                target_xpos=target_xpos.contiguous(),
                qpos_seed=qpos_seed,
                return_all_solutions=return_all_solutions,
                search_mode=search_mode,
            )


def get_iiwa_urdf_path() -> str:
    """Locate iiwa resources in a source checkout or an installed wheel."""
    path = Path(__file__).resolve().parent / "urdf" / "lbr_iiwa7_r800.urdf"
    if not path.is_file():
        from srs_kinematics_warp_assets import URDF_PATH

        path = URDF_PATH
    return str(path)


class KUKAiiwa7R800Solver(SRSSolver):
    """Ready-to-use iiwa 7 R800 specialization matching the bundled URDF."""

    def __init__(
        self,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        cfg: SRSSolverCfg | None = None,
        **kwargs,
    ):
        if cfg is None:
            cfg = SRSSolverCfg()
            lower = np.deg2rad([-170, -120, -170, -120, -170, -120, -175])
            cfg.qpos_limits = np.column_stack((lower, -lower))
        if not getattr(cfg, "urdf_path", None):
            cfg.urdf_path = get_iiwa_urdf_path()
        if not getattr(cfg, "joint_names", None):
            cfg.joint_names = [f"joint{i + 1}" for i in range(7)]
        super().__init__(cfg=cfg, num_envs=num_envs, device=device, **kwargs)


KUKAiiwaSolver = KUKAiiwa7R800Solver


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
    # Example usage
    cfg = SRSSolverCfg()
    solver = SRSSolver(cfg, num_envs=1, device="cpu")

    # First compute FK with a known joint configuration
    test_qpos = np.array(
        [
            -0.12123752,
            -0.24020538,
            0.62607642,
            -1.25394088,
            0.16017357,
            -1.06349794,
            -0.56939818,
        ]
    )

    # Compute FK manually using the DH parameters
    impl = solver.impl
    target_xpos_fk = impl._get_fk(test_qpos)

    print("FK Result (with zero joint angles):")
    print(target_xpos_fk)

    # Now use the FK result as input to IK
    target_xpos = torch.from_numpy(target_xpos_fk).float().unsqueeze(0)
    qpos_seed = torch.zeros((1, 7))

    print("Running IK with the FK target...")
    success, qpos = solver.get_ik(target_xpos, qpos_seed)
    print("IK Success:", success)
    print("IK Solution (rad):", qpos)

    qpos_np = qpos.detach().cpu().numpy().reshape(-1, 7)[0]
    target_pose_ = impl._get_fk(qpos_np)
    print("FK Result (with small non-zero joint angles):")
    print(target_pose_)
