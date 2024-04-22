import os
import sys

import cvxopt
import mujoco_py
import transforms3d.quaternions
from gym import spaces
from mujoco_py import functions as mjf
from scipy.linalg import cho_solve, cho_factor

sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.envs.ho_im4 import HandObjMimic4

from uhc.utils.transformation import (
    quat_mul_vec,
)
from uhc.utils.math_utils import *
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
)


class HandObjMimicTest(HandObjMimic4):
    def __init__(self, cfg, init_expert_seq, model_path, data_specs, mode="train"):
        self.expert_window = init_expert_seq
        self.w_size = cfg.future_w_size
        self.expert_index = [i for i in range(self.w_size + 1)]

        HandObjMimic4.__init__(self, cfg, init_expert_seq, model_path, data_specs, mode="train")

    def insert_new_frame(self, frame):
        expire_idx = self.expert_index[0]
        self.expert_window[expire_idx] = frame
        for i in range(self.w_size + 1):
            self.expert_index[i] = (self.expert_index[i] + 1) % (self.w_size + 1)

    def get_expert_attr(self, attr, ind):
        assert ind <= self.w_size
        e_idx = self.expert_index[ind]
        return self.expert_window[e_idx][attr].copy()

    def get_expert_hand_qpos(self, delta_t=0):
        expert_qpos = self.get_expert_attr("hand_dof_seq", delta_t)
        return expert_qpos

    def get_expert_hand_qvel(self, delta_t=0):
        expert_vel = self.get_expert_attr("hand_dof_vel_seq", delta_t)
        return expert_vel

    def get_expert_hand_root_pose(self, delta_t=0):
        expert_root_pos = self.get_expert_attr("body_pos_seq", delta_t)[0]
        expert_root_quat = self.get_expert_attr("body_quat_seq", delta_t)[0]
        return np.concatenate([expert_root_pos, expert_root_quat])

    def get_expert_obj_pose(self, delta_t=0):
        obj_pose = self.get_expert_attr("obj_pose_seq", delta_t)
        return obj_pose

    def get_expert_obj_vel(self, delta_t=0):
        obj_vel = self.get_expert_attr("obj_vel_seq", delta_t)
        obj_ang_vel = self.get_expert_attr("obj_angle_vel_seq", delta_t)
        return np.concatenate([obj_vel, obj_ang_vel])

    def get_expert_wbody_pos(self, delta_t=0):
        # world joint position
        wbpos = self.get_expert_attr("body_pos_seq", delta_t)
        wbpos = wbpos.reshape(-1, 3).flatten()
        return wbpos

    def get_expert_wbody_quat(self, delta_t=0):
        # world joint position
        wbquat = self.get_expert_attr("body_quat_seq", delta_t)
        wbquat = wbquat.reshape(-1, 4).flatten()
        return wbquat

    def reset_model(self, tracking=False):
        cfg = self.cc_cfg
        init_hand_pose_exp = self.get_expert_hand_qpos()
        init_hand_vel_exp = self.get_expert_hand_qvel()
        init_obj_pose_exp = self.get_expert_obj_pose()
        init_obj_vel_exp = self.get_expert_obj_vel()

        init_pose = np.concatenate([init_hand_pose_exp, init_obj_pose_exp])
        init_vel = np.concatenate([init_hand_vel_exp, init_obj_vel_exp])

        self.set_state(init_pose, init_vel)

        return self.get_obs()
