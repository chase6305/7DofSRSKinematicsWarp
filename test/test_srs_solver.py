import numpy as np
import torch

from srs_solver import SRSSolver, SRSSolverCfg


def test_srs_solver_ik():
    cfg = SRSSolverCfg()
    cfg.dh_params = np.array(
        [
            [0.34, -np.pi / 2, 0, 0],
            [0, np.pi / 2, 0, 0],
            [0.4, np.pi / 2, 0, 0],
            [0, -np.pi / 2, 0, 0],
            [0.4, -np.pi / 2, 0, 0],
            [0, np.pi / 2, 0, 0],
            [0.126, 0, 0, np.pi / 2],
        ]
    )
    cfg.link_lengths = np.array([0.34, 0.4, 0.4, 0.126])
    cfg.rotation_directions = [1, 1, 1, 1, 1, 1, 1]
    cfg.T_b_ob = np.eye(4)
    cfg.T_e_oe = np.eye(4)
    cfg.num_samples = 120
    cfg.sort_ik = True
    cfg.ik_nearest_weight = [1, 1, 1, 1, 1, 1, 1]
    cfg.joint_names = [f"joint_{i + 1}" for i in range(7)]
    cfg.urdf_path = "./urdf/lbr_iiwa7_r800.urdf"

    # 创建 solver
    solver = SRSSolver(cfg, num_envs=1, device="cpu")

    test_qpos_fk = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    impl = solver.impl
    target_pose = impl._get_fk(test_qpos_fk)

    target_xpos = torch.from_numpy(target_pose).float().unsqueeze(0)

    qpos_seed = torch.zeros((1, 7), dtype=torch.float32)

    success, qpos = solver.get_ik(target_xpos, qpos_seed, return_all_solutions=False)
    assert bool(success.detach().cpu().numpy()[0])

    qpos_np = qpos.detach().cpu().numpy().reshape(-1, 7)[0]
    target_pose_ = impl._get_fk(qpos_np)
    assert np.allclose(target_pose_, target_pose, atol=1e-3)


if __name__ == "__main__":
    test_srs_solver_ik()
