# 日志设置
import logging
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import torch
import warp as wp

try:
    import coloredlogs

    coloredlogs.install(
        level="INFO", fmt="%(asctime)s,%(msecs)03d %(levelname)s %(message)s"
    )
except ImportError:  # pragma: no cover - standard logging is a valid fallback
    logging.basicConfig(level=logging.INFO)


# 添加 logger 实例，兼容 log_warning/log_error/log_debug
class _LoggerWrapper:
    def __init__(self, logger):
        self._logger = logger

    def log_warning(self, msg):
        self._logger.warning(msg)

    def log_error(self, msg):
        self._logger.error(msg)

    def log_debug(self, msg):
        self._logger.debug(msg)


logger = _LoggerWrapper(logging.getLogger(__name__))

from dataclasses import dataclass, field

from warp_kernel import (
    check_success_kernel,
    compute_fk_kernel,
    compute_ik_kernel,
    nearest_ik_kernel,
    sort_ik_kernel,
    transform_pose_kernel,
)

__all__ = [
    "ArmKineParams",
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


def standardize_device_string(device: str | torch.device) -> str:
    """Standardize the device string for Warp compatibility.

    Args:
        device (Union[str, torch.device]): The device specification.

    Returns:
        str: The standardized device string.
    """
    if isinstance(device, str):
        device_str = device
    else:
        device_str = str(device)

    if device_str == "cuda":
        device_str = "cuda:0"

    return device_str


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
        # self.d_list = np.array([0.0, 0.0, 0.260, 0.0, 0.166, 0.098, 0.0])
        self.link_lengths = np.array([0.34, 0.4, 0.4, 0.126])

        # helpers: create DH rows and clamp limits
        def dh_row(d, alpha, a, theta):
            return [d, alpha, a, theta]

        def deg2rad_list(list_of_pairs):
            return np.deg2rad(np.array(list_of_pairs, dtype=float))

        self.T_b_ob = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.T_e_oe = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # fmt: off
        dh = [
            dh_row(self.link_lengths[0],  -np.pi / 2,  0.0,  0.0),
            dh_row(0.0,                    np.pi / 2,  0.0,  0.0),
            dh_row(self.link_lengths[1],   np.pi / 2,  0.0,  0.0),
            dh_row(0.0,                   -np.pi / 2,  0.0,  0.0),
            dh_row(self.link_lengths[2],  -np.pi / 2,  0.0,  0.0),
            dh_row(0.0,                    np.pi / 2,  0.0,  0.0),
            dh_row(self.link_lengths[3],   0.0,        0.0,  0.0),
        ]
        # fmt: on

        limits = [
            [-180.0, 180.0],
            [-180.0, 180.0],
            [-180.0, 180.0],
            [-180.0, 180.0],
            [-180.0, 180.0],
            [-180.0, 180.0],
            [-180.0, 180.0],
        ]
        rotation_directions = np.array([1, 1, 1, 1, 1, 1, 1])

        # finalize arrays
        self.dh_params = np.array(dh, dtype=float)
        self.qpos_limits = deg2rad_list(limits)
        self.rotation_directions = rotation_directions

        # sanity checks
        assert self.dh_params.shape == (7, 4), "dh_params must be shape (7,4)"
        assert self.qpos_limits.shape == (7, 2), "qpos_limits must be shape (7,2)"


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
    qpos_limits: np.ndarray = field(default_factory=lambda: np.zeros((7, 2)))
    tcp: np.ndarray = field(default_factory=lambda: np.eye(4))
    ik_nearest_weight: list = field(default_factory=lambda: [1.0] * 7)
    num_samples: int = 100
    sort_ik: bool = True

    def __post_init__(self):
        # Populate from ArmKineParams when fields are not provided
        if len(self.dh_params) == 0 or len(self.link_lengths) == 0:
            kine = ArmKineParams()
            if len(self.dh_params) == 0:
                self.dh_params = kine.dh_params.tolist()
            if len(self.link_lengths) == 0:
                self.link_lengths = kine.link_lengths.tolist()
            if len(self.rotation_directions) == 0:
                self.rotation_directions = kine.rotation_directions.tolist()
            if np.allclose(self.T_b_ob, np.eye(4)):
                self.T_b_ob = kine.T_b_ob
            if np.allclose(self.T_e_oe, np.eye(4)):
                self.T_e_oe = kine.T_e_oe
            if np.allclose(self.qpos_limits, np.zeros((7, 2))):
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
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if len(self.ik_nearest_weight) != 7:
            raise ValueError("ik_nearest_weight must contain seven values")


@dataclass(frozen=True)
class SRSConfiguration:
    """Discrete shoulder/elbow/wrist branch and continuous arm angle."""

    shoulder: int
    elbow: int
    wrist: int
    redundancy: float

    def __post_init__(self):
        for name in ("shoulder", "elbow", "wrist"):
            if getattr(self, name) not in (-1, 1):
                raise ValueError(f"{name} must be -1 or 1")
        if not np.isfinite(self.redundancy):
            raise ValueError("redundancy must be finite")


class _BaseSRSSolverImpl:
    """Base implementation for the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        # Initialize configuration and device
        self.cfg = cfg
        self.cfg.validate()
        self.device = device
        self.dofs = 7
        self.dh_params = np.asarray(cfg.dh_params)
        self.tcp_xpos = np.eye(4)
        # Initialize transformation matrices
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


class _CPUSRSSolverImpl(_BaseSRSSolverImpl):
    """CPU implementation of the SRS inverse kinematics solver."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        super().__init__(cfg, device)

    def _parse_params(self):
        super()._parse_params()

        # Generate all possible configuration combinations for shoulder, elbow, and wrist.
        # Each configuration is represented by a vector of three elements, each being +1 or -1.
        # This covers all 8 possible sign combinations for the three joints.
        self.configs = [
            np.array([x, y, z]) for x, y, z in product([1.0, -1.0], repeat=3)
        ]

        # Generate a set of elbow angles sampled uniformly from -π to π.
        # The number of samples is determined by self.cfg.num_samples.
        # These angles are used for searching possible IK solutions.
        self.elbow_angles = torch.linspace(
            -torch.pi, torch.pi, self.cfg.num_samples, device=self.device
        )

        # Convert ik_nearest_weight to a tensor for efficient computation.
        self.ik_nearest_weight_tensor = torch.tensor(
            self.cfg.ik_nearest_weight, dtype=torch.float32, device=self.device
        )

    def _get_fk(self, target_joint: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics (FK) for a given joint state.

        Args:
            target_joint (np.ndarray): Joint angles (shape: [7,]).

        Returns:
            np.ndarray: 4x4 transformation matrix representing the end-effector pose.
        """
        # Initialize pose as identity matrix
        pose = np.eye(4)

        # Iterate through the DH parameters and compute the transformation
        for i in range(self.dh_params.shape[0]):
            d = self.dh_params[i, 0]
            alpha = self.dh_params[i, 1]
            a = self.dh_params[i, 2]
            theta = self.dh_params[i, 3]

            # Add joint angle contribution if within bounds
            if i < target_joint.size:
                theta += target_joint[i] * self.cfg.rotation_directions[i]

            # Compute the transformation matrix for this joint
            T = self._dh_transform(d, alpha, a, theta)
            pose = pose @ T

        # Apply additional transformations: user frame, base, and tool center point (TCP)
        pose = (
            self.cfg.T_b_ob
            @ pose
            @ self.cfg.T_e_oe  # End-effector-to-observed-end-effector transform
            @ self.tcp_xpos  # Tool center point transform
        )

        return pose

    def _get_arm_angle(self, qpos: np.ndarray) -> float:
        """Measure the SRS elbow-plane angle for one joint configuration."""
        joints = np.asarray(qpos, dtype=np.float64)
        target = self._get_fk(joints)
        local_target = (
            self.T_b_ob_inv_np @ target @ self.tcp_inv_np @ self.T_e_oe_inv_np
        )
        elbow = -1 if joints[3] < 0.0 else 1
        _, reference_rotation, _ = self._compute_reference_plane(local_target, elbow)
        if reference_rotation is None:
            raise ValueError("arm angle is undefined at this configuration")
        actual_rotation = np.eye(3)
        for index in range(3):
            d, alpha, a, theta = self.dh_params[index]
            transform = self._dh_transform(
                d,
                alpha,
                a,
                theta + joints[index] * self.rotation_directions_np[index],
            )
            actual_rotation = actual_rotation @ transform[:3, :3]
        wrist = local_target[:3, 3] - local_target[:3, :3] @ np.array(
            [0.0, 0.0, self.dh_params[-1, 0]]
        )
        axis = wrist - np.array([0.0, 0.0, self.link_lengths_np[0]])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            raise ValueError("arm angle is undefined at this configuration")
        axis /= axis_norm
        projections = [
            reference_rotation[:, index] - axis * axis.dot(reference_rotation[:, index])
            for index in range(3)
        ]
        column = int(np.argmax([np.linalg.norm(value) for value in projections]))
        reference = projections[column] / np.linalg.norm(projections[column])
        actual = actual_rotation[:, column]
        actual -= axis * axis.dot(actual)
        actual /= np.linalg.norm(actual)
        return float(
            np.arctan2(axis.dot(np.cross(reference, actual)), reference.dot(actual))
        )

    def _calculate_arm_joint_angles(
        self,
        P26: np.ndarray,
        elbow_config: int,
        joints: np.ndarray,
        link_lengths: np.ndarray,
    ) -> bool:
        """
        Calculate joint angles based on the position vector P26.

        Args:
            P26 (np.ndarray): Vector from shoulder to wrist.
            elbow_config (int): Elbow configuration (+1 or -1).
            joints (np.ndarray): Joint angles to be updated.
            link_lengths (np.ndarray): Link lengths of the robot.

        Returns:
            bool: True if successful, False otherwise.
        """
        _, d_se, d_ew = link_lengths[:3]

        norm_P26 = np.linalg.norm(P26)
        if (
            norm_P26 > d_se + d_ew + 1e-10
            or norm_P26 < abs(d_se - d_ew) - 1e-10
            or norm_P26 < 1e-12
        ):
            logger.log_debug("Specified pose outside reachable workspace.")
            return False

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
        if abs(elbow_cos_angle) > 1.0 + 1e-10:
            logger.log_debug("Elbow singularity. End effector at limit.")
            return False

        joints[3] = elbow_config * np.arccos(np.clip(elbow_cos_angle, -1.0, 1.0))

        euclidean_norm = np.hypot(P26[0], P26[1])
        if euclidean_norm > 1e-10:
            joints[0] = np.arctan2(P26[1], P26[0])
        else:
            joints[0] = 0

        angle_phi_cos = (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26)
        angle_phi = np.arccos(np.clip(angle_phi_cos, -1.0, 1.0))
        joints[1] = np.arctan2(euclidean_norm, P26[2]) + elbow_config * angle_phi

        return True

    def _dh_transform(
        self, d: float, alpha: float, a: float, theta: float
    ) -> np.ndarray:
        """
        Compute the Denavit-Hartenberg transformation matrix.

        Args:
            d (float): Link offset.
            alpha (float): Link twist.
            a (float): Link length.
            theta (float): Joint angle.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

        # fmt: off
        return np.array(
            [
                [cos_theta,  -sin_theta * cos_alpha, sin_theta * sin_alpha,  a * cos_theta],
                [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha,  a * sin_theta],
                [0,          sin_alpha,              cos_alpha,              d],
                [0, 0, 0, 1],
            ]
        )
        # fmt: on

    def _skew(self, vector: np.ndarray) -> np.ndarray:
        """
        Compute the skew-symmetric matrix of a vector.

        Args:
            vector (np.ndarray): Input vector (3,).

        Returns:
            np.ndarray: Skew-symmetric matrix (3x3).
        """
        return np.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def _compute_reference_plane(
        self, target_pose: np.ndarray, elbow_config: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Calculate the reference plane vector, rotation matrix, and joint values.

        Args:
            target_pose (np.ndarray): Transformed target pose (4x4).
            elbow_config (int): Elbow configuration (+1 or -1).

        Returns:
            tuple: (plane_normal, base_to_elbow_rotation, joint_angles) or (None, None, None) if failed.
        """
        dh_params = self.dh_params
        link_lengths = self.cfg.link_lengths

        P_target = target_pose[:3, 3]
        P02 = np.array([0, 0, link_lengths[0]])
        P67 = np.array([0, 0, dh_params[6, 0]])
        P06 = P_target - target_pose[:3, :3] @ P67
        P26 = P06 - P02

        joint_angles = np.zeros(7)
        if not self._calculate_arm_joint_angles(
            P26, elbow_config, joint_angles, link_lengths
        ):
            return None, None, None

        T34_v = self._dh_transform(
            dh_params[3, 0], dh_params[3, 1], dh_params[3, 2], joint_angles[3]
        )
        P34_v = T34_v[:3, 3]

        norm_P34_P02 = np.linalg.norm(P34_v - P02)
        if norm_P34_P02 > 1e-6:
            v1 = (P34_v - P02) / norm_P34_P02
        else:
            v1 = np.zeros_like(P34_v - P02)
        v2 = (P06 - P02) / np.linalg.norm(P06 - P02)
        plane_normal = np.cross(v1, v2)

        base_to_elbow_rotation = np.eye(3)
        for i in range(3):
            T = self._dh_transform(
                dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], joint_angles[i]
            )
            base_to_elbow_rotation = base_to_elbow_rotation @ T[:3, :3]

        return plane_normal, base_to_elbow_rotation, joint_angles

    def _process_all_solutions(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns all valid IK solutions (optionally sorted).

        Args:
            ik_qpos_tensor (torch.Tensor): The IK joint position tensor.
            qpos_seed (torch.Tensor): The seed joint position tensor.
            valid_mask (torch.Tensor): The mask indicating valid solutions.
            success_tensor (torch.Tensor): The tensor indicating success of IK solutions.

        Returns:
            torch.Tensor: The success tensor.
            torch.Tensor: The IK solutions tensor (sorted if specified).
        """
        if self.cfg.sort_ik:
            weighted_diff = (
                ik_qpos_tensor - qpos_seed.unsqueeze(1)
            ) * self.ik_nearest_weight_tensor
            distances = torch.norm(weighted_diff, dim=2)
            distances[~valid_mask] = float("inf")
            sorted_indices = torch.argsort(distances, dim=1)
            sorted_ik_qpos_tensor = torch.gather(
                ik_qpos_tensor, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 7)
            )
            return success_tensor, sorted_ik_qpos_tensor
        else:
            return success_tensor, ik_qpos_tensor

    def _process_single_solution(
        self,
        ik_qpos_tensor: torch.Tensor,
        qpos_seed: torch.Tensor,
        valid_mask: torch.Tensor,
        success_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the nearest valid IK solution (optionally sorted).

        Args:
            ik_qpos_tensor (torch.Tensor): The IK joint position tensor.
            qpos_seed (torch.Tensor): The seed joint position tensor.
            valid_mask (torch.Tensor): The mask indicating valid solutions.
            success_tensor (torch.Tensor): The tensor indicating success of IK solutions.

        Returns:
            torch.Tensor: The success tensor.
            torch.Tensor: The nearest valid IK solution tensor.
        """
        num_targets = ik_qpos_tensor.shape[0]
        if self.cfg.sort_ik:
            weighted_diff = (
                ik_qpos_tensor - qpos_seed.unsqueeze(1)
            ) * self.ik_nearest_weight_tensor
            distances = torch.norm(weighted_diff, dim=2)
            mask = success_tensor.unsqueeze(1) & valid_mask
            distances[~mask] = float("inf")
            nearest_indices = torch.argmin(distances, dim=1)
            nearest_solutions = torch.zeros(
                (num_targets, 7), dtype=qpos_seed.dtype, device=self.device
            )
            has_solution = distances.min(dim=1).values != float("inf")
            if has_solution.any():
                nearest_solutions[has_solution] = ik_qpos_tensor[
                    torch.arange(num_targets)[has_solution],
                    nearest_indices[has_solution],
                ]
            return success_tensor, nearest_solutions.unsqueeze(1)
        else:
            # Return first solution only
            return success_tensor, ik_qpos_tensor[:, :1, :]

    def _get_each_ik(
        self, target_pose: np.ndarray, nsparam: float, config: np.ndarray
    ) -> tuple[bool, np.ndarray | None]:
        """
        Computes the inverse kinematics for a given target pose, normalization parameter, and configuration.

        Args:
            target_pose (np.ndarray): 4x4 target pose matrix.
            nsparam (float): Normalization parameter (angle).
            config (np.ndarray): Configuration index.

        Returns:
            bool: Success flag.
            np.ndarray: List of joint solutions (7) or None if no solution is found.
        """
        # Validate the target pose matrix
        target_pose = np.array(target_pose)
        if target_pose.ndim == 3 and target_pose.shape[0] == 1:
            target_pose = target_pose[0]  # Extract the first matrix
        if target_pose.shape != (4, 4):
            logger.log_error(
                f"Invalid xpos shape: {target_pose.shape}, expected (4,4)."
            )
            return False, None

        shoulder_config, elbow_config, wrist_config = config[0], config[1], config[2]

        dof = self.dofs
        joints_output = np.zeros(dof)

        # Extract parameters
        dh_params = self.dh_params
        link_lengths = self.cfg.link_lengths
        rotation_directions = self.cfg.rotation_directions

        # Transform target pose
        target_xpos = (
            self.T_b_ob_inv_np @ target_pose @ self.tcp_inv_np @ self.T_e_oe_inv_np
        )
        P_target = target_xpos[:3, 3]
        R_target = target_xpos[:3, :3]
        P02 = np.array([0, 0, link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, dh_params[6, 0]])  # Hand to end-effector
        P06 = P_target - R_target @ P67
        P26 = P06 - P02

        # Calculate joint angles
        joints = np.zeros(dof)
        if not self._calculate_arm_joint_angles(
            P26, elbow_config, joints, link_lengths
        ):
            return False, None

        # Calculate transformations
        T34 = self._dh_transform(
            dh_params[3, 0], dh_params[3, 1], dh_params[3, 2], joints[3]
        )
        R34 = T34[:3, :3]

        # Calculate reference plane
        V_v_to_sew, R03_o, _ = self._compute_reference_plane(target_xpos, config[1])
        if V_v_to_sew is None:
            return False, None

        # Calculate shoulder joint rotation matrices
        usw = P26 / np.linalg.norm(P26)
        skew_usw = self._skew(usw)
        angle_psi = nsparam
        s_psi = np.sin(angle_psi)
        c_psi = np.cos(angle_psi)

        # Calculate rotation matrix R03
        A_s = skew_usw @ R03_o
        B_s = -skew_usw @ skew_usw @ R03_o
        C_s = (usw[:, None] @ usw[None, :]) @ R03_o
        R03 = A_s * s_psi + B_s * c_psi + C_s

        # Calculate shoulder joint angles
        angle1 = np.arctan2(R03[1, 1] * shoulder_config, R03[0, 1] * shoulder_config)
        angle2 = np.arccos(R03[2, 1]) * shoulder_config
        angle3 = np.arctan2(-R03[2, 2] * shoulder_config, -R03[2, 0] * shoulder_config)

        # Calculate wrist joint angles
        A_w = R34.T @ A_s.T @ R_target
        B_w = R34.T @ B_s.T @ R_target
        C_w = R34.T @ C_s.T @ R_target
        R47 = A_w * s_psi + B_w * c_psi + C_w

        angle5 = np.arctan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        angle6 = np.arccos(R47[2, 2]) * wrist_config
        angle7 = np.arctan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

        joints_output[0] = (angle1 - dh_params[0, 3]) * rotation_directions[0]
        joints_output[1] = (angle2 - dh_params[1, 3]) * rotation_directions[1]
        joints_output[2] = (angle3 - dh_params[2, 3]) * rotation_directions[2]
        joints_output[3] = (joints[3] - dh_params[3, 3]) * rotation_directions[3]
        joints_output[4] = (angle5 - dh_params[4, 3]) * rotation_directions[4]
        joints_output[5] = (angle6 - dh_params[5, 3]) * rotation_directions[5]
        joints_output[6] = (angle7 - dh_params[6, 3]) * rotation_directions[6]

        # Check if the calculated joint angles are within the limits
        in_range = (joints_output >= self.qpos_limits_np[:, 0]) & (
            joints_output <= self.qpos_limits_np[:, 1]
        )

        if not np.all(in_range):
            return False, None

        return True, joints_output

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose using CPU.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad).
            return_all_solutions: Whether to return all solutions. Default is False.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        num_targets = target_xpos.shape[0]
        # Validate and normalize qpos_seed
        if qpos_seed is None:
            qpos_seed = torch.zeros(
                (target_xpos.shape[0], 7), dtype=torch.float32, device=self.device
            )

        # Prepare to collect results
        max_possible_solutions = len(self.elbow_angles) * len(self.configs)
        all_solutions = np.zeros(
            (num_targets, max_possible_solutions, 7), dtype=np.float32
        )
        solution_counts = np.zeros(num_targets, dtype=np.int32)

        target_numpy = target_xpos.detach().cpu().numpy()
        seed_numpy = qpos_seed.detach().cpu().numpy()

        def solve_target(target_idx):
            xpos = target_numpy[target_idx]
            # Match the reference solver's stationary-target fast path.  It is
            # important for control loops and also handles straight-arm
            # singularities where the elbow plane is undefined.
            if not return_all_solutions and np.allclose(
                self._get_fk(seed_numpy[target_idx]), xpos, atol=1e-7, rtol=0.0
            ):
                return target_idx, seed_numpy[target_idx][None, :]
            if not return_all_solutions:
                try:
                    seed = seed_numpy[target_idx]
                    seed_config = np.where(seed[[1, 3, 5]] < 0.0, -1.0, 1.0)
                    seed_arm_angle = self._get_arm_angle(seed)
                    local_configs = [seed_config]
                    local_configs.extend(
                        config
                        for config in self.configs
                        if not np.array_equal(config, seed_config)
                    )
                    step = np.deg2rad(5.0)
                    offsets = [0.0]
                    for layer in range(1, 5):
                        offsets.extend((-layer * step, layer * step))
                    for local_config in local_configs:
                        for offset in offsets:
                            local_success, local_qpos = self._get_each_ik(
                                xpos, seed_arm_angle + offset, local_config
                            )
                            if local_success and np.allclose(
                                self._get_fk(local_qpos), xpos, atol=1e-4, rtol=0.0
                            ):
                                return target_idx, local_qpos[None, :]
                except (ValueError, FloatingPointError):
                    pass
            sol_idx = 0
            target_solutions = np.zeros((max_possible_solutions, 7), dtype=np.float32)
            for psi in self.elbow_angles:
                for config in self.configs:
                    success, qpos = self._get_each_ik(xpos, psi.item(), config)
                    if success:
                        fk_xpos = self._get_fk(qpos)
                        if np.allclose(fk_xpos, xpos, atol=1e-4):
                            target_solutions[sol_idx] = qpos
                            sol_idx += 1
            return target_idx, target_solutions[:sol_idx]

        # The CPU fallback intentionally stays simple; the Warp device path is
        # the high-throughput implementation.
        for target_idx in range(num_targets):
            _, solutions = solve_target(target_idx)
            solution_counts[target_idx] = len(solutions)
            all_solutions[target_idx, : len(solutions)] = solutions

        has_solution = solution_counts > 0
        if not any(has_solution):
            logger.log_warning(
                f"Failed to calculate IK solutions.\n"
                f"Target pose: {target_xpos}\nSeed: {qpos_seed}"
            )
            return (
                torch.zeros(num_targets, dtype=torch.bool, device=self.device),
                torch.zeros(
                    (
                        num_targets,
                        max_possible_solutions if return_all_solutions else 1,
                        7,
                    ),
                    dtype=qpos_seed.dtype,
                    device=self.device,
                ),
            )
        max_solutions = max_possible_solutions

        # Convert results to tensors
        ik_qpos_tensor = torch.zeros(
            (num_targets, max_solutions, 7),
            dtype=qpos_seed.dtype,
            device=self.device,
        )
        for target_idx in range(num_targets):
            count = solution_counts[target_idx]
            if count > 0:
                ik_qpos_tensor[target_idx, :count] = torch.from_numpy(
                    all_solutions[target_idx, :count]
                ).to(self.device, dtype=qpos_seed.dtype)

        valid_mask = (
            torch.arange(max_solutions, device=self.device)[None, :]
            < torch.as_tensor(solution_counts, device=self.device)[:, None]
        )
        success_tensor = torch.from_numpy(has_solution).to(self.device)
        if return_all_solutions:
            return self._process_all_solutions(
                ik_qpos_tensor, qpos_seed, valid_mask, success_tensor
            )
        else:
            return self._process_single_solution(
                ik_qpos_tensor, qpos_seed, valid_mask, success_tensor
            )


class _WarpSRSSolverImpl(_BaseSRSSolverImpl):
    """Warp implementation for CUDA or CPU devices."""

    def __init__(self, cfg: SRSSolverCfg, device: torch.device):
        super().__init__(cfg, device)

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

        # Generate all possible configuration combinations for shoulder, elbow, and wrist.
        # Each configuration is represented by a vector of three elements, each being +1 or -1.
        # This covers all 8 possible sign combinations for the three joints.
        self.configs = [wp.vec3(x, y, z) for x, y, z in product([1.0, -1.0], repeat=3)]
        self.configs_wp = wp.array(
            self.configs, dtype=wp.vec3, device=standardize_device_string(self.device)
        )

        # Generate a set of elbow angles sampled uniformly from -π to π.
        # The number of samples is determined by self.cfg.num_samples.
        # These angles are used for searching possible IK solutions.
        joint_reference_limits = [-wp.pi, wp.pi]
        self.elbow_angles = np.linspace(
            joint_reference_limits[0], joint_reference_limits[1], self.cfg.num_samples
        ).tolist()

        # Convert elbow angles to Warp array for CUDA computation.
        self.elbow_angles_wp = wp.array(
            self.elbow_angles,
            dtype=float,
            device=standardize_device_string(self.device),
        )
        self._workspace_key = None
        self._workspace = {}

    def _ensure_workspace(self, num_targets: int, num_combinations: int) -> dict:
        """Reuse large device buffers across control-loop calls of equal shape."""
        key = (num_targets, num_combinations)
        if key == self._workspace_key:
            return self._workspace
        device = standardize_device_string(self.device)
        self._workspace_key = key
        self._workspace = {
            "xpos": wp.empty(num_targets, dtype=wp.mat44, device=device),
            "qpos": wp.empty(num_combinations * 7, dtype=float, device=device),
            "success": wp.empty(num_combinations, dtype=int, device=device),
        }
        return self._workspace

    def _sort_ik_solutions(
        self, qpos_out_wp, success_wp, qpos_seed, num_targets, num_configs, num_angles
    ):
        """
        Sort IK solutions based on weighted distance.

        Args:
            qpos_out_wp: Warp array of IK solutions (shape: [num_targets * num_configs * num_angles, 7]).
            success_wp: Warp array of validity flags (shape: [num_targets * num_configs * num_angles]).
            qpos_seed: Warp array of seed positions (shape: [num_targets, 7]).
            num_targets: Number of targets.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[wp.array, wp.array]: Sorted IK solutions and their validity flags.
        """
        N = num_targets
        N_SOL = num_configs * num_angles
        DOF = 7

        sorted_ik_solutions = wp.zeros(
            N * N_SOL * DOF, dtype=float, device=standardize_device_string(self.device)
        )
        sorted_ik_valid_flags = wp.zeros(
            N * N_SOL, dtype=int, device=standardize_device_string(self.device)
        )
        distances = wp.zeros(
            N * N_SOL, dtype=float, device=standardize_device_string(self.device)
        )
        indices = wp.zeros(
            N * N_SOL, dtype=int, device=standardize_device_string(self.device)
        )

        wp.launch(
            kernel=sort_ik_kernel,
            dim=num_targets,
            inputs=[
                qpos_out_wp,
                success_wp,
                qpos_seed,
                wp.array(
                    self.cfg.ik_nearest_weight,
                    dtype=float,
                    device=standardize_device_string(self.device),
                ),
                distances,
                indices,
                N_SOL,
            ],
            outputs=[
                sorted_ik_solutions,
                sorted_ik_valid_flags,
            ],
            device=standardize_device_string(self.device),
        )
        return sorted_ik_solutions, sorted_ik_valid_flags

    def _nearest_ik_solution(
        self, qpos_out_wp, success_wp, qpos_seed, num_targets, num_configs, num_angles
    ):
        """
        Find the nearest valid IK solution for each target pose.

        Selects the IK solution closest to the seed configuration among all valid solutions.

        Args:
            qpos_out_wp: IK solutions array of shape [num_targets * num_configs * num_angles, 7]
            success_wp: Validity flags array of shape [num_targets * num_configs * num_angles]
            qpos_seed: Seed configurations array of shape [num_targets, 7]
            num_targets: Number of target poses
            num_configs: Number of IK configurations
            num_angles: Number of sampling angles

        Returns:
            Tuple[wp.array, wp.array]:
                - Nearest IK solutions array of shape [num_targets, 7]
                - Validity flags array of shape [num_targets] indicating solution feasibility
        """
        N = num_targets
        N_SOL = num_configs * num_angles
        DOF = 7

        nearest_ik_solutions = wp.zeros(
            N * DOF, dtype=float, device=standardize_device_string(self.device)
        )
        nearest_ik_valid_flags = wp.zeros(
            N, dtype=int, device=standardize_device_string(self.device)
        )

        wp.launch(
            kernel=nearest_ik_kernel,
            dim=num_targets,
            inputs=[
                qpos_out_wp,
                success_wp,
                qpos_seed.flatten(),
                wp.array(
                    self.cfg.ik_nearest_weight,
                    dtype=float,
                    device=standardize_device_string(self.device),
                ),
                N_SOL,
            ],
            outputs=[
                nearest_ik_solutions,
                nearest_ik_valid_flags,
            ],
            device=standardize_device_string(self.device),
        )
        return nearest_ik_solutions, nearest_ik_valid_flags

    def _process_all_solutions(
        self,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        qpos_seed: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process and return all valid IK solutions.

        Args:
            qpos_out_wp: Warp array of IK solutions.
            success_wp: Warp array of success flags.
            qpos_seed: Seed joint positions.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flags and all valid joint positions.
        """
        num_per_target = num_configs * num_angles
        solutions = wp.to_torch(qpos_out_wp).view(num_targets, num_per_target, 7)
        valid = wp.to_torch(success_wp).view(num_targets, num_per_target).bool()
        if self.cfg.sort_ik:
            weights = torch.as_tensor(
                self.cfg.ik_nearest_weight, dtype=solutions.dtype, device=self.device
            )
            distances = torch.linalg.vector_norm(
                (solutions - qpos_seed[:, None, :]) * weights, dim=2
            ).masked_fill(~valid, torch.inf)
            order = distances.argsort(dim=1)
            solutions = torch.gather(solutions, 1, order[:, :, None].expand(-1, -1, 7))
            valid = torch.gather(valid, 1, order)
        return valid.any(dim=1), solutions.masked_fill(~valid[:, :, None], 0.0)

    def _process_single_solution(
        self,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        qpos_seed: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process and return the nearest valid IK solution for each target.

        Args:
            qpos_out_wp: Warp array of IK solutions.
            success_wp: Warp array of success flags.
            qpos_seed: Seed joint positions.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flags and nearest valid joint positions.
        """
        num_per_target = num_configs * num_angles
        solutions = wp.to_torch(qpos_out_wp).view(num_targets, num_per_target, 7)
        valid = wp.to_torch(success_wp).view(num_targets, num_per_target).bool()
        if self.cfg.sort_ik:
            weights = torch.as_tensor(
                self.cfg.ik_nearest_weight, dtype=solutions.dtype, device=self.device
            )
            distances = torch.linalg.vector_norm(
                (solutions - qpos_seed[:, None, :]) * weights, dim=2
            ).masked_fill(~valid, torch.inf)
            index = distances.argmin(dim=1)
        else:
            index = valid.to(torch.int32).argmax(dim=1)
        rows = torch.arange(num_targets, device=self.device)
        has_solution = valid.any(dim=1)
        nearest = solutions[rows, index].masked_fill(~has_solution[:, None], 0.0)
        return has_solution, nearest[:, None, :]

    def _check_success_flags(
        self,
        success_wp: wp.array,
        num_targets: int,
        num_configs: int,
        num_angles: int,
    ) -> torch.Tensor:
        """
        Check success flags for IK solutions.

        Args:
            success_wp: Warp array of success flags.
            num_targets: Number of target poses.
            num_configs: Number of configurations.
            num_angles: Number of angles.

        Returns:
            torch.Tensor: Success flags as a boolean tensor.
        """
        num_solutions = num_configs * num_angles
        success_flags_wp = wp.empty(
            num_targets, dtype=int, device=standardize_device_string(self.device)
        )
        wp.launch(
            kernel=check_success_kernel,
            dim=num_targets,
            inputs=[
                success_wp,
                num_solutions,
            ],
            outputs=[
                success_flags_wp,
            ],
            device=standardize_device_string(self.device),
        )
        return wp.to_torch(success_flags_wp).bool().to(self.device)

    def _compute_ik_solutions(
        self,
        xpos_wp: wp.array,
        qpos_out_wp: wp.array,
        success_wp: wp.array,
        num_combinations: int,
    ) -> None:
        """
        Compute IK solutions using the provided combinations.

        Args:
            xpos_wp: Transformed target poses.
            qpos_out_wp: Output array for joint positions.
            success_wp: Output array for success flags.
            num_combinations: Total number of combinations to process.
        """
        # Launch kernel to compute IK solutions
        wp.launch(
            kernel=compute_ik_kernel,
            dim=num_combinations,
            inputs=(
                xpos_wp,
                self.elbow_angles_wp,
                self.qpos_limits_wp,
                self.configs_wp,
                self.dh_params_wp,
                self.link_lengths_wp,
                self.rotation_directions_wp,
            ),
            outputs=[success_wp, qpos_out_wp],
            device=standardize_device_string(self.device),
        )

    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor,
        return_all_solutions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inverse kinematics (IK) for the given target pose.

        Args:
            target_xpos: Target end-effector pose (4x4).
            qpos_seed: Initial joint positions (rad).
            return_all_solutions: Whether to return all solutions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Success flag and joint positions.
        """
        # Prepare inputs
        target_xpos = target_xpos.to(self.device)
        target_xpos = target_xpos.view(-1, 4, 4)
        target_xpos_wp = wp.from_torch(target_xpos, dtype=wp.mat44)

        num_targets = target_xpos_wp.shape[0]
        num_configs = len(self.configs)
        num_angles = len(self.elbow_angles)
        num_combinations = num_targets * num_configs * num_angles
        workspace = self._ensure_workspace(num_targets, num_combinations)

        # transform pose
        xpos_wp = workspace["xpos"]
        wp.launch(
            kernel=transform_pose_kernel,
            dim=target_xpos_wp.shape[0],
            inputs=[
                target_xpos_wp,
                self.T_b_ob_inv_wp,
                self.T_e_oe_inv_wp,
                self.tcp_inv_wp,
            ],
            outputs=[xpos_wp],
            device=standardize_device_string(self.device),
        )

        # Define configurations and angles
        if qpos_seed is None:
            qpos_seed = torch.zeros(
                (target_xpos.shape[0], 7),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            qpos_seed = qpos_seed.to(self.device, dtype=torch.float32).view(-1, 7)
            # TODO: Currently, full-space sampling is used to temporarily address situations
            # where joint space discontinuities or solution failures occur in different user scenarios.
            # Future plans include reducing the sampling space and adjusting the configuration.
            #
            # self.configs = [wp.vec3(*np.sign(qpos_seed[[1, 3, 5]].cpu().numpy()))]

        # Prepare output arrays
        qpos_out_wp = workspace["qpos"]
        success_wp = workspace["success"]

        # Compute IK solutions
        self._compute_ik_solutions(xpos_wp, qpos_out_wp, success_wp, num_combinations)

        # Device-side masking below handles the all-failed case without a host
        # synchronization or an extra success-count kernel.
        if return_all_solutions:
            return self._process_all_solutions(
                qpos_out_wp,
                success_wp,
                qpos_seed,
                num_targets,
                num_configs,
                num_angles,
            )
        return self._process_single_solution(
            qpos_out_wp,
            success_wp,
            qpos_seed,
            num_targets,
            num_configs,
            num_angles,
        )


class SRSSolver:
    r"""SRS inverse kinematics (IK) controller.

    This controller implements SRS inverse kinematics using various methods for
    computing the inverse of the Jacobian matrix.
    """

    def __init__(
        self,
        cfg: SRSSolverCfg,
        num_envs: int,
        device: str | torch.device | None,
        use_warp: bool | None = None,
        **kwargs,
    ):
        r"""Initializes the SRS kinematics solver.

            This constructor sets up the kinematics solver using SRS methods,
            allowing for efficient computation of robot kinematics based on
            the specified URDF model.

        Args:
            cfg: The configuration for the solver.
            num_envs (int): The number of environments for the solver.
            device (str, optional): The device to use for the solver (e.g., "cpu" or "cuda").
            **kwargs: Additional keyword arguments passed to the base solver.

        """
        # super().__init__(cfg=cfg, num_envs=num_envs, device=device, **kwargs)  # 移除父类初始化

        self.cfg = cfg
        self.cfg.validate()
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.use_warp = self.device.type == "cuda" if use_warp is None else use_warp
        if self.device.type == "cuda" and not self.use_warp:
            raise ValueError("CUDA execution requires the Warp backend")
        if self.use_warp:
            _ensure_writable_warp_cache()

        self.urdf_path = getattr(cfg, "urdf_path", None)
        self.joint_names = getattr(cfg, "joint_names", None)
        self.end_link_name = getattr(cfg, "end_link_name", None)
        self.root_link_name = getattr(cfg, "root_link_name", None)

        # Degrees of freedom
        self.dofs = 7

        self.dof = len(self.joint_names) if self.joint_names else self.dofs

        # Weight for nearest neighbor search in IK (Inverse Kinematics) algorithms
        if hasattr(cfg, "ik_nearest_weight") and cfg.ik_nearest_weight is not None:
            if len(cfg.ik_nearest_weight) != self.dof:
                logging.getLogger(__name__).error(
                    f"Length of ik_nearest_weight ({len(cfg.ik_nearest_weight)}) does not match the number of DOF ({self.dof})."
                )
            self.ik_nearest_weight = torch.tensor(
                cfg.ik_nearest_weight, dtype=torch.float32, device=self.device
            )
        else:
            self.ik_nearest_weight = torch.ones(
                self.dof, dtype=torch.float32, device=self.device
            )

        self.tcp_xpos = np.eye(4)

        # Tool Center Point (TCP) position
        self.tcp_xpos = np.eye(4)

        # 兼容无 pk_serial_chain 情况
        if hasattr(self, "pk_serial_chain"):
            fk_dict = self.pk_serial_chain.forward_kinematics(
                th=np.zeros(7), end_only=False
            )
            root_tf = fk_dict[next(iter(fk_dict))]
            self.root_base_xpos = root_tf.get_matrix().cpu().numpy()
        else:
            self.root_base_xpos = np.eye(4)

        # Initialize joint position limits from cfg
        qpos_limits = np.asarray(cfg.qpos_limits)  # shape (7, 2)
        self.lower_qpos_limits = torch.tensor(
            qpos_limits[:, 0], dtype=torch.float32, device=self.device
        )
        self.upper_qpos_limits = torch.tensor(
            qpos_limits[:, 1], dtype=torch.float32, device=self.device
        )

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
        tcp = np.asarray(torch.as_tensor(transform).detach().cpu(), dtype=np.float64)
        if tcp.shape != (4, 4) or not np.all(np.isfinite(tcp)):
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(tcp[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
            raise ValueError("transform must be homogeneous")
        self.cfg.tcp = tcp.copy()
        if self.use_warp:
            self.impl = _WarpSRSSolverImpl(self.cfg, self.device)
            self._geometry_impl = _CPUSRSSolverImpl(self.cfg, torch.device("cpu"))
        else:
            self.impl = _CPUSRSSolverImpl(self.cfg, self.device)
            self._geometry_impl = self.impl
        self._update_impl_qpos_limits()

    def get_arm_angle(self, qpos: np.ndarray | torch.Tensor) -> float:
        """Measure elbow-plane rotation around the shoulder-to-wrist axis."""
        joints = np.asarray(torch.as_tensor(qpos).detach().cpu(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("qpos must be a finite array of shape (7,)")
        return self._geometry_impl._get_arm_angle(joints)

    def get_configuration(self, qpos: np.ndarray | torch.Tensor) -> SRSConfiguration:
        """Return the current S/E/W branch and continuous arm angle."""
        joints = np.asarray(torch.as_tensor(qpos).detach().cpu(), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("qpos must be a finite array of shape (7,)")

        def sign(value):
            return -1 if value < 0.0 else 1

        return SRSConfiguration(
            sign(joints[1]),
            sign(joints[3]),
            sign(joints[5]),
            self.get_arm_angle(joints),
        )

    def solve_configuration(
        self,
        target_xpos: np.ndarray | torch.Tensor,
        configuration: SRSConfiguration,
    ) -> tuple[bool, torch.Tensor | None]:
        """Solve one explicitly selected branch at an exact arm angle."""
        target = np.asarray(
            torch.as_tensor(target_xpos).detach().cpu(), dtype=np.float64
        )
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("target_xpos must be a finite 4x4 matrix")
        branch = np.array(
            [configuration.shoulder, configuration.elbow, configuration.wrist],
            dtype=np.float64,
        )
        success, joints = self._geometry_impl._get_each_ik(
            target, configuration.redundancy, branch
        )
        if not success:
            return False, None
        actual = self._geometry_impl._get_fk(joints)
        if not np.allclose(actual, target, atol=1e-4, rtol=0.0):
            return False, None
        return True, torch.as_tensor(joints, dtype=torch.float32, device=self.device)

    def _update_impl_qpos_limits(self):
        qpos_limits = torch.vstack([self.lower_qpos_limits, self.upper_qpos_limits]).T
        self.impl.qpos_limits_np = qpos_limits.cpu().numpy()
        if self.use_warp:
            self.impl.qpos_limits_wp = wp.array(
                self.impl.qpos_limits_np,
                dtype=wp.vec2,
                device=standardize_device_string(self.device),
            )
        if hasattr(self, "_geometry_impl"):
            self._geometry_impl.qpos_limits_np = self.impl.qpos_limits_np.copy()

    def update_with_robot_limit(self, robot_qpos_limits):
        limits = np.asarray(robot_qpos_limits, dtype=np.float32)
        if limits.shape != (7, 2):
            raise ValueError("robot_qpos_limits must have shape (7, 2)")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("each lower joint limit must not exceed its upper limit")
        self.lower_qpos_limits = torch.as_tensor(
            limits[:, 0], dtype=torch.float32, device=self.device
        )
        self.upper_qpos_limits = torch.as_tensor(
            limits[:, 1], dtype=torch.float32, device=self.device
        )
        self._update_impl_qpos_limits()

    def get_fk(self, qpos: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute FK for input shape ``(7,)`` or ``(N, 7)``.

        Warp execution launches one thread per robot.  The result always
        has shape ``(N, 4, 4)`` on the solver device.
        """
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        if qpos_tensor.ndim == 1:
            qpos_tensor = qpos_tensor.unsqueeze(0)
        if qpos_tensor.ndim != 2 or qpos_tensor.shape[1] != 7:
            raise ValueError("qpos must have shape (7,) or (N, 7)")
        if not bool(torch.isfinite(qpos_tensor).all()):
            raise ValueError("qpos must contain only finite values")

        if not self.use_warp:
            poses = np.stack(
                [self.impl._get_fk(row) for row in qpos_tensor.cpu().numpy()], axis=0
            )
            return torch.as_tensor(poses, dtype=qpos_tensor.dtype, device=self.device)

        warp_device = standardize_device_string(self.device)
        joint_angles = wp.from_torch(qpos_tensor.contiguous().flatten())
        poses = wp.empty(qpos_tensor.shape[0], dtype=wp.mat44, device=warp_device)
        success = wp.empty(qpos_tensor.shape[0], dtype=int, device=warp_device)
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
            outputs=[poses, success],
            device=warp_device,
        )
        return wp.to_torch(poses).view(-1, 4, 4)

    def get_jacobian(
        self, qpos: np.ndarray | torch.Tensor, step: float = 1e-4
    ) -> torch.Tensor:
        """Return geometric Jacobians, batching all FK differences in one launch."""
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be positive and finite")
        joints = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        squeeze = joints.ndim == 1
        if squeeze:
            joints = joints.unsqueeze(0)
        if joints.ndim != 2 or joints.shape[1] != 7:
            raise ValueError("qpos must have shape (7,) or (N, 7)")
        batch = joints.shape[0]
        offsets = torch.zeros((14, 7), dtype=joints.dtype, device=self.device)
        ids = torch.arange(7, device=self.device)
        offsets[2 * ids, ids] = step
        offsets[2 * ids + 1, ids] = -step
        samples = (joints[:, None, :] + offsets[None, :, :]).reshape(-1, 7)
        poses = self.get_fk(samples).reshape(batch, 7, 2, 4, 4)
        plus, minus = poses[:, :, 0], poses[:, :, 1]
        linear = (plus[:, :, :3, 3] - minus[:, :, :3, 3]) / (2.0 * step)
        center_rotation = self.get_fk(joints)[:, :3, :3]
        rotation_rate = (plus[:, :, :3, :3] - minus[:, :, :3, :3]) / (2.0 * step)
        skew_rate = rotation_rate @ center_rotation[:, None].transpose(-1, -2)
        angular = torch.stack(
            (skew_rate[:, :, 2, 1], skew_rate[:, :, 0, 2], skew_rate[:, :, 1, 0]),
            dim=-1,
        )
        jacobian = torch.cat((linear, angular), dim=-1).transpose(1, 2)
        return jacobian[0] if squeeze else jacobian

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
        preferred = torch.as_tensor(
            preferred_velocity, dtype=jacobian.dtype, device=self.device
        )
        if preferred.ndim == 1:
            preferred = preferred.unsqueeze(0)
        if preferred.shape != (jacobian.shape[0], 7):
            raise ValueError("preferred_velocity must have shape (7,) or (N, 7)")
        projector = (
            torch.eye(7, dtype=jacobian.dtype, device=self.device).expand(
                jacobian.shape[0], -1, -1
            )
            - torch.linalg.pinv(jacobian) @ jacobian
        )
        velocity = (projector @ preferred.unsqueeze(-1)).squeeze(-1)
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
        target_xpos = torch.as_tensor(
            target_xpos, dtype=torch.float32, device=self.device
        )
        if target_xpos.ndim == 2:
            target_xpos = target_xpos.unsqueeze(0)
        if target_xpos.ndim != 3 or target_xpos.shape[1:] != (4, 4):
            raise ValueError("target_xpos must have shape (4, 4) or (N, 4, 4)")
        if not bool(torch.isfinite(target_xpos).all()):
            raise ValueError("target_xpos must contain only finite values")
        if qpos_seed is not None:
            qpos_seed = torch.as_tensor(
                qpos_seed, dtype=torch.float32, device=self.device
            )
            if qpos_seed.ndim == 1:
                qpos_seed = qpos_seed.unsqueeze(0)
            if qpos_seed.shape != (target_xpos.shape[0], 7):
                raise ValueError("qpos_seed must have shape (7,) or (N, 7)")
            if not bool(torch.isfinite(qpos_seed).all()):
                raise ValueError("qpos_seed must contain only finite values")
        return self.impl.get_ik(
            target_xpos=target_xpos,
            qpos_seed=qpos_seed,
            return_all_solutions=return_all_solutions,
            **kwargs,
        )


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
            cfg.urdf_path = str(
                Path(__file__).resolve().parent / "urdf" / "lbr_iiwa7_r800.urdf"
            )
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

    from IPython import embed

    embed(header="IK Solution Debug")
