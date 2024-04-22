import argparse
import os
import sys

import mujoco
import numpy as np
import transforms3d.quaternions

from uhc.utils.transforms import *

sys.path.append(os.getcwd())

import mujoco_py
import cvxopt
import qpsolvers
import transforms3d as t3d
from gym import spaces
from scipy.linalg import cho_solve, cho_factor

from uhc.data_loaders.dataset_grab import DatasetGRAB
from uhc.utils.config_utils.handmimic_config import Config
from uhc.khrylib.utils import *
from uhc.utils.math_utils import *

from uhc.envs.mujoco3_env import Mujoco3Env


class MotionData:
    def __init__(self):
        self.hand_kps = []
        self.obj_pose = []
        self.contact_frames = []
        self.contact_force = []
        self.contact_body = []
        self.compensate_ft = []
        self.obj_vel = []
        self.obj_acc = []


# Environment for mujoco3
class HandObjMimic4New(Mujoco3Env):
    def __init__(self, cfg, expert_seq, model_xml, data_specs, mode="train"):
        self.sim_step = cfg.sim_step
        Mujoco3Env.__init__(self, model_xml, self.sim_step)

        # self.model = (mujoco.MjModel)(self.model)

        # load cfg
        self.cc_cfg = cfg
        self.set_cam_first = set()
        self.pos_diff_thresh = cfg.pos_diff_thresh
        self.rot_diff_thresh = cfg.rot_diff_thresh
        self.jpos_diff_thresh = cfg.jpos_diff_thresh
        self.obj_pos_diff_thresh = cfg.obj_pos_diff_thresh
        self.obj_rot_diff_thresh = cfg.obj_rot_diff_thresh
        self.random_start = cfg.random_start
        self.obs_type = cfg.get("obs_type", 0)
        self.noise_future_pose = cfg.noise_future_pose
        self.noise_scale = 1.0 if mode == 'train' else 0.0
        self.fk_data = mujoco.MjData(self.model)
        self.w_size = cfg.future_w_size
        self.start_ind = 0

        # load hand model
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # read some params from model
        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.hand_qpos_dim = self.model.nq - 7
        self.hand_qvel_dim = self.model.nv - 6
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.all_body_names = [mujoco.mj_id2name(self.model, 1, i) for i in range(self.model.nbody)]
        self.all_geom_names = [mujoco.mj_id2name(self.model, 5, i) for i in range(self.model.ngeom)]
        self.hand_body_names = [name for name in self.all_body_names if name.startswith("link")]
        self.hand_body_idx = [mujoco.mj_name2id(self.model, 1, name) for name in self.hand_body_names]
        self.obj_body_idx = self.hand_body_idx[-1] + 1
        self.hand_geom_range = [-1, -1]
        for i, name in enumerate(self.all_geom_names):
            if name.startswith('robot0:') and self.hand_geom_range[0] == -1:
                self.hand_geom_range[0] = i
            if (not name.startswith('robot0:')) and self.hand_geom_range[0] != -1:
                self.hand_geom_range[1] = i - 1
                break
        self.hand_geom_num = self.hand_geom_range[1] - self.hand_geom_range[0] + 1
        self.obj_geom_range = [-1, -1]
        self.obj_geom_range[0] = self.hand_geom_range[1] + 1
        for i in range(self.obj_geom_range[0], len(self.all_geom_names)):
            name = self.all_geom_names[i]
            if not name.startswith('C_'):
                self.obj_geom_range[1] = i - 1
                break
        print("Hand geom range: ", self.hand_geom_range)
        print("Object geom idx: ", self.obj_geom_range)
        self.hand_mass = 0
        for idx in self.hand_body_idx:
            self.hand_mass += self.model.body_mass[idx]
        print("Hand mass", self.hand_mass)
        self.hand_body_num = len(self.hand_body_idx)
        self.obj_contacts = np.array([])
        self.torque_lim = cfg.torque_lim
        self.jkp = cfg.jkp
        self.jkd = cfg.jkd
        self.joint_upper_limit = self.model.jnt_range[:-1, 1]
        self.joint_lower_limit = self.model.jnt_range[:-1, 0]
        self.base_pose = (self.joint_upper_limit + self.joint_lower_limit) / 2.0
        self.ctrl_scale = self.joint_upper_limit - self.base_pose
        self.ctrl_scale[6:] *= 1.2  # make control range larger to enable control ability around the joint limits

        # load expert
        self.expert = expert_seq
        if isinstance(self.expert, dict):
            self.expert_len = expert_seq["hand_dof_seq"].shape[0]
        else:
            self.expert_len = 0

        self.mode = mode
        self.end_reward = 0.0
        self.curr_vf = None  # storing current vf
        self.curr_torque = None  # Strong current applied torque at each joint

        # setup viewer
        self.viewer = self._get_viewer('hand') if cfg.render else None
        self._viewers = {}

        self.reset()
        self.set_action_spaces()
        self.set_obs_spaces()

        # motion data
        self.motion_data = MotionData()

    def set_mode(self, mode):
        self.mode = mode

    def set_expert(self, expert_seq):
        self.expert = expert_seq
        self.expert_len = expert_seq["hand_dof_seq"].shape[0]

    def set_action_spaces(self):
        cfg = self.cc_cfg
        self.vf_dim = 0
        self.meta_pd_dim = 0
        body_id_list = self.model.geom_bodyid.tolist()

        if cfg.residual_force:
            self.vf_dim = 6

        if cfg.meta_pd:
            self.meta_pd_dim = 2 * self.sim_step
        elif cfg.meta_pd_joint:
            self.meta_pd_dim = 2 * self.jkp.shape[0]

        control_dim = self.ndof
        if self.cc_cfg.grot_type == 'quat':
            control_dim += 1
        self.action_dim = control_dim + self.vf_dim + self.meta_pd_dim
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float32,
        )

    def set_obs_spaces(self):
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def get_obs(self):
        if self.obs_type == 3:
            obs = self.get_full_obs_v4(self.w_size)
        else:
            obs = self.get_full_obs_v5(self.w_size)
        return obs

    def get_full_obs_v4(self, w_size=1):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        cur_root_pose = self.get_hand_root_pose()
        obs = []

        # ######## Add noise to future pose #########
        noise_qpos, noise_root_pose, noise_joint_pos, noise_obj_pose = None, None, None, None
        if self.noise_future_pose:
            future_qpos_arr = []
            future_obj_pose_arr = []
            for idx in range(1, w_size):
                future_qpos_arr.append(self.get_expert_hand_qpos(delta_t=1 + idx))
                future_obj_pose_arr.append(self.get_expert_obj_pose(delta_t=1 + idx))
            noise_qpos, noise_root_pose, noise_joint_pos, noise_obj_pose = \
                self.get_noise_pose(future_qpos_arr, future_obj_pose_arr)

        ######## current hand pose #########
        hand_rot_mat = transforms3d.quaternions.quat2mat(cur_root_pose[3:])
        obs.append(hand_rot_mat[:, 0:2].flatten('F'))
        obs.append(qpos[6:self.hand_qpos_dim])

        ######## target hand pose relative to current pose #########
        for idx in range(0, w_size):
            if self.noise_future_pose and idx > 0:
                obs.append(noise_qpos[idx - 1] - qpos[6:self.hand_qpos_dim])
            else:
                target_hand_qpos = self.get_expert_hand_qpos(delta_t=1 + idx)
                obs.append(target_hand_qpos[6:] - qpos[6:self.hand_qpos_dim])

        ################ hand vels ################
        # vel
        obs.append(qvel[6:self.hand_qvel_dim])
        # transform global vel to local coordinates
        local_root_vel = transform_vec(qvel[:3], cur_root_pose[3:])
        local_root_ang_vel = transform_vec(qvel[3:6], cur_root_pose[3:])
        obs.append(local_root_vel)
        obs.append(local_root_ang_vel)

        ######### relative root position and orientation #########
        for idx in range(0, w_size):
            if self.noise_future_pose and idx > 0:
                target_root_pose = noise_root_pose[idx - 1]
            else:
                target_root_pose = self.get_expert_hand_root_pose(delta_t=1 + idx)
            rel_root_pos = target_root_pose[:3] - cur_root_pose[:3]
            rel_root_pos = transform_vec(rel_root_pos, cur_root_pose[3:])
            rel_root_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), target_root_pose[3:])
            rel_root_rot = transforms3d.quaternions.quat2mat(rel_root_rot)
            obs.append(rel_root_pos)
            obs.append(rel_root_rot[:, :2].flatten('F'))

        ############# target/difference joint positions ############
        curr_jpos = self.get_wbody_pos().reshape(-1, 3)
        r_jpos = curr_jpos[1:] - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, cur_root_pose[3:])  # body frame position
        obs.append(r_jpos.ravel())
        for idx in range(0, w_size):
            if self.noise_future_pose and idx > 0:
                target_jpos = noise_joint_pos[idx - 1]
            else:
                target_jpos = self.get_expert_wbody_pos(delta_t=1 + idx).reshape(-1, 3)

            # translate to body frame (zero-out root)
            diff_jpos = target_jpos[1:] - curr_jpos[1:]
            diff_jpos = transform_vec_batch(diff_jpos, cur_root_pose[3:])
            obs.append(diff_jpos.ravel())

        ############# object pose  ############
        obj_root_pose = qpos[self.hand_qpos_dim:]
        obj_pos_rel_hand = obj_root_pose[:3] - cur_root_pose[:3]
        obj_pos_rel_hand = transform_vec(obj_pos_rel_hand, cur_root_pose[3:])
        obj_rot_rel_hand = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), obj_root_pose[3:])
        obj_rot_rel_hand = transforms3d.quaternions.quat2mat(obj_rot_rel_hand)
        obs.append(obj_pos_rel_hand)
        obs.append(obj_rot_rel_hand[:, 0:2].flatten('F'))

        ############# object vel  ############
        obj_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim:self.hand_qvel_dim + 3], cur_root_pose[3:])
        obj_ang_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim + 3:self.hand_qvel_dim + 6], cur_root_pose[3:])
        obs.append(obj_vel_rel_hand)
        obs.append(obj_ang_vel_rel_hand)

        ############# object target pose relative to current pose ############
        for idx in range(0, w_size):
            if self.noise_future_pose and idx > 0:
                target_obj_pose = noise_obj_pose[idx - 1]
            else:
                target_obj_pose = self.get_expert_obj_pose(delta_t=1 + idx)
            rel_obj_pos = target_obj_pose[:3] - obj_root_pose[:3]
            rel_obj_pos = transform_vec(rel_obj_pos, cur_root_pose[3:])
            rel_obj_rot = quaternion_multiply(target_obj_pose[3:], quaternion_inverse(obj_root_pose[3:]))
            rel_obj_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), rel_obj_rot)
            rel_obj_rot = transforms3d.quaternions.quat2mat(rel_obj_rot)
            obs.append(rel_obj_pos)
            obs.append(rel_obj_rot[:, 0:2].flatten('F'))

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v5(self, w_size=1):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        cur_root_pose = self.get_hand_root_pose()
        obs = []

        ######## current hand pose #########
        hand_rot_mat = transforms3d.quaternions.quat2mat(cur_root_pose[3:])
        obs.append(hand_rot_mat[:, 0:2].flatten())
        obs.append(qpos[6:self.hand_qpos_dim])

        ######## target hand pose relative to current pose #########
        for idx in range(0, w_size):
            target_hand_qpos = self.get_expert_hand_qpos(delta_t=1 + idx)
            obs.append(target_hand_qpos[6:] - qpos[6:self.hand_qpos_dim])

        ################ hand vels ################
        # vel
        obs.append(qvel[6:self.hand_qvel_dim])
        # transform global vel to local coordinates
        local_root_vel = transform_vec(qvel[:3], cur_root_pose[3:])
        local_root_ang_vel = transform_vec(qvel[3:6], cur_root_pose[3:])
        obs.append(local_root_vel)
        obs.append(local_root_ang_vel)

        ######### relative root position and orientation #########
        for idx in range(0, w_size):
            target_root_pose = self.get_expert_hand_root_pose(delta_t=1 + idx)
            rel_root_pos = target_root_pose[:3] - cur_root_pose[:3]
            rel_root_pos = transform_vec(rel_root_pos, cur_root_pose[3:])
            rel_root_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), target_root_pose[3:])
            rel_root_rot = transforms3d.quaternions.quat2mat(rel_root_rot)
            obs.append(rel_root_pos)
            obs.append(rel_root_rot[:, :2].flatten())

        ############# target/difference joint positions ############
        curr_jpos = self.get_wbody_pos().reshape(-1, 3)
        r_jpos = curr_jpos[1:] - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, cur_root_pose[3:])  # body frame position
        obs.append(r_jpos.ravel())
        for idx in range(0, w_size):
            target_jpos = self.get_expert_wbody_pos(delta_t=1 + idx).reshape(-1, 3)

            # translate to body frame (zero-out root)
            diff_jpos = target_jpos[1:] - curr_jpos[1:]
            diff_jpos = transform_vec_batch(diff_jpos, cur_root_pose[3:])
            obs.append(diff_jpos.ravel())

        ############# object pose  ############
        obj_root_pose = qpos[self.hand_qpos_dim:]
        obj_pos_rel_hand = obj_root_pose[:3] - cur_root_pose[:3]
        obj_pos_rel_hand = transform_vec(obj_pos_rel_hand, cur_root_pose[3:])
        obj_rot_rel_hand = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), obj_root_pose[3:])
        obj_rot_rel_hand = transforms3d.quaternions.quat2mat(obj_rot_rel_hand)
        obs.append(obj_pos_rel_hand)
        obs.append(obj_rot_rel_hand[:, 0:2].flatten())

        ############# object vel  ############
        obj_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim:self.hand_qvel_dim + 3], cur_root_pose[3:])
        obj_ang_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim + 3:self.hand_qvel_dim + 6], cur_root_pose[3:])
        obs.append(obj_vel_rel_hand)
        obs.append(obj_ang_vel_rel_hand)

        ############# object target pose relative to current pose ############
        for idx in range(0, w_size):
            target_obj_pose = self.get_expert_obj_pose(delta_t=1 + idx)
            rel_obj_pos = target_obj_pose[:3] - obj_root_pose[:3]
            rel_obj_pos = transform_vec(rel_obj_pos, cur_root_pose[3:])
            rel_obj_rot = quaternion_multiply(target_obj_pose[3:], quaternion_inverse(obj_root_pose[3:]))
            rel_obj_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), rel_obj_rot)
            rel_obj_rot = transforms3d.quaternions.quat2mat(rel_obj_rot)
            obs.append(rel_obj_pos)
            obs.append(rel_obj_rot[:, 0:2].flatten())

        obs = np.concatenate(obs)
        return obs

    def get_noise_pose(self, qpos_arr, obj_pose_arr):
        frame_num = len(qpos_arr)
        noise_qpos = []
        noise_hand_root = []
        noise_hand_body_pos = []
        noise_obj_pose = []
        for i in range(frame_num):
            # add noise to translation
            qpos = qpos_arr[i].copy()
            obj_pose = obj_pose_arr[i].copy()
            qpos[:3] += 0.003 * self.noise_scale * np.random.randn(3)
            obj_pose[:3] += 0.003 * self.noise_scale * np.random.randn(3)

            # add noise to rotation
            root_quat = t3d.euler.euler2quat(qpos[3], qpos[4], qpos[5])
            root_quat = root_quat + 0.02 * self.noise_scale * np.random.randn(4)
            qpos[3:6] = t3d.euler.quat2euler(root_quat / np.linalg.norm(root_quat))
            obj_pose[3:] += 0.02 * self.noise_scale * np.random.randn(4)
            obj_pose[3:] = obj_pose[3:] / np.linalg.norm(obj_pose[3:])

            # add noise to loco pose
            qpos[6:] = qpos[6:] + 0.05 * self.noise_scale * np.random.randn(len(qpos[6:]))

            # forward kinematics to get noise keypoints
            self.fk_data.qpos[:] = np.concatenate([qpos, obj_pose])
            mujoco.mj_forward(self.model, self.fk_data)
            hand_kp = self.fk_data.xpos[self.hand_body_idx].copy()
            root_quat = self.fk_data.xquat[self.hand_body_idx[0]].copy()

            noise_qpos.append(qpos[6:])
            noise_hand_root.append(np.concatenate([qpos[:3], root_quat]))
            noise_hand_body_pos.append(hand_kp)
            noise_obj_pose.append(obj_pose)
        return noise_qpos, noise_hand_root, noise_hand_body_pos, noise_obj_pose

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv

        M = np.zeros(nv * nv)
        M.resize(nv, nv)
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M = M[:self.hand_qvel_dim, :self.hand_qvel_dim]
        C = self.data.qfrc_bias.copy()[:self.hand_qvel_dim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

    def compute_torque(self, ctrl, i_iter=0):
        cfg = self.cc_cfg
        dt = self.model.opt.timestep
        qpos = self.get_hand_qpos()
        qvel = self.get_hand_qvel()
        ref_qpos = self.get_expert_hand_qpos()
        target_pos = np.zeros_like(qpos)

        ctrl_pos = ctrl[:3]
        ctrl_rot = ctrl[3:7] if cfg.grot_type == 'quat' else ctrl[3:6]
        ctrl_loco = ctrl[7:self.ndof + 1] if cfg.grot_type == 'quat' else ctrl[6:self.ndof]

        target_pos[0:3] = ref_qpos[0:3] + 0.1 * ctrl_pos

        if cfg.pd_type == 'base':
            # target_pos[0:3] = self.base_pose[0:3] + self.ctrl_scale[0:3] * ctrl_pos
            target_pos[6:] = self.base_pose[6:] + self.ctrl_scale[6:] * ctrl_loco
        else:
            # target_pos[0:3] = qpos[0:3] + self.ctrl_scale[0:3] * ctrl_pos
            target_pos[6:] = ref_qpos[6:] + self.ctrl_scale[6:] * ctrl_loco

        if self.cc_cfg.grot_type == 'euler':
            # target_pos[3:6] = (self.base_pose[3:6] if cfg.pd_type == 'base' else qpos[3:6]) \
            #                   + self.ctrl_scale[3:6] * ctrl_rot
            target_pos[3:6] = ref_qpos[3:6] + 0.3 * ctrl_rot
        else:
            ctrl_joint = ctrl[:self.ndof + 1]
            grot_quat = ctrl_joint[3:7] / np.linalg.norm(ctrl_joint[3:7])
            grot_mat = transforms3d.quaternions.quat2mat(grot_quat)
            if cfg.pd_type == 'base':
                base_mat = transforms3d.euler.euler2mat(self.base_pose[3], self.base_pose[4], self.base_pose[5])
            else:
                base_mat = transforms3d.euler.euler2mat(qpos[3], qpos[4], qpos[5])
            target_grot_mat = np.matmul(grot_mat, base_mat)
            target_pos[3:6] = transforms3d.euler.mat2euler(target_grot_mat)

        if self.mode == 'test':
            self.target_hand_pose = target_pos

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])

        if cfg.meta_pd:
            meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
            curr_jkp = self.jkp * (meta_pds[i_iter] + 1)
            curr_jkd = self.jkd * (meta_pds[i_iter + self.sim_step] + 1)
            # if flags.debug:
            # import ipdb; ipdb.set_trace()
            # print((meta_pds[i_iter + self.sim_iter] + 1), (meta_pds[i_iter] + 1))
        elif cfg.meta_pd_joint:
            num_jts = self.jkp.shape[0]
            meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
            curr_jkp = self.jkp * (meta_pds[:num_jts] + 1)
            curr_jkd = self.jkd * (meta_pds[num_jts:] + 1)
        else:
            curr_jkp = self.jkp
            curr_jkd = self.jkd

        k_p[:] = curr_jkp
        k_d[:] = curr_jkd
        qpos_err = qpos[:] + qvel[:] * dt - target_pos
        qvel_err = qvel

        # align all orientation err to [-pi, pi]
        rot_qpos_err = qpos_err[3:].copy()
        while np.any(rot_qpos_err > np.pi):
            rot_qpos_err[rot_qpos_err > np.pi] -= 2 * np.pi
        while np.any(rot_qpos_err < -np.pi):
            rot_qpos_err[rot_qpos_err < -np.pi] += 2 * np.pi
        qpos_err[3:] = rot_qpos_err[:]

        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -curr_jkp * qpos_err - curr_jkd * qvel_err
        return torque

    def rfc_obj(self):
        obj_mc = self.get_obj_qpos()[:3]
        qfrc = self.data.qfrc_applied.copy()

        mujoco.mj_applyFT(
            self.model,
            self.data,
            self.obj_vf,
            self.obj_vt,
            obj_mc,
            self.obj_body_idx,
            qfrc,
        )
        self.data.qfrc_applied[:] = qfrc

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl = action

        old_obj_vel = self.data.qvel[-6:].copy()
        old_geom_xpos = self.data.geom_xpos.copy()
        old_geom_rot = self.data.geom_xmat.copy()
        old_geom_rot = old_geom_rot.reshape(old_geom_rot.shape[0], 3, 3)

        self.contact_frame_arr = [[] for i in range(self.hand_geom_num)]
        self.contact_num_count = np.zeros(self.hand_geom_num)

        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl, i_iter=i)
            elif cfg.action_type == "torque":
                torque = ctrl * self.a_scale * 100
            torque = np.clip(torque, -self.torque_lim, self.torque_lim)

            self.data.ctrl[:] = torque

            """ Add extra force to balance hand gravity """
            qfrc = np.zeros(self.data.qfrc_applied.shape)
            mujoco.mj_applyFT(
                self.model,
                self.data,
                np.array([0, 0, self.hand_mass * 9.8]),
                np.zeros(3).astype(np.float64),
                self.data.geom_xpos[2],
                3,
                qfrc,
            )
            self.data.qfrc_applied[:] = qfrc

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                self.rfc_obj()

            """ Record contact """
            self.record_contact()

            mujoco.mj_step(self.model, self.data)

            if self.mode == 'test':
                self.compute_data_frame()

            if self.viewer:
                self.viewer.render()

        delta_t = n_frames * self.model.opt.timestep
        self.obj_avg_acc = (self.data.qvel[-6:] - old_obj_vel) / delta_t
        self.geom_avg_vel = (self.data.geom_xpos - old_geom_xpos) / delta_t
        geom_rot = self.data.geom_xmat.copy()
        geom_rot = geom_rot.reshape(old_geom_rot.shape[0], 3, 3)
        geom_rot_diff = np.matmul(geom_rot, np.transpose(old_geom_rot, [0, 2, 1]))
        self.geom_avg_ang_vel = matrix_to_axis_angle(torch.Tensor(geom_rot_diff)).numpy() / delta_t

        # average contact
        self.avg_cps, self.avg_cp_geom, self.cp_ts = self.classify_contact()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def classify_contact(self):
        contact_frames = []
        contact_geom_idx = []
        contact_ts_count = []
        for i in range(self.hand_geom_num):
            if self.contact_num_count[i] == 0:
                continue
            all_contact = np.stack(self.contact_frame_arr[i])
            mean_contact_frame = np.mean(all_contact, axis=0)

            diff = all_contact[:, 0:3][:, None] - all_contact[:, 0:3]
            distances = np.linalg.norm(diff, axis=2)
            max_distance = np.max(distances)
            # print(max_distance * 1000)

            mean_contact_frame[3:6] = mean_contact_frame[3:6] / np.linalg.norm(mean_contact_frame[3:6])
            avg_n = mean_contact_frame[3:6]

            # compute tangential direction
            if abs(np.dot(avg_n, np.array([1, 0, 0]))) >= 1e-5:
                mean_contact_frame[6:9] = np.cross(avg_n, np.array([1, 0, 0]))
            else:
                mean_contact_frame[6:9] = np.cross(avg_n, np.array([0, 1, 0]))
            mean_contact_frame[6:9] = mean_contact_frame[6:9] / np.linalg.norm(mean_contact_frame[6:9])
            mean_contact_frame[9:] = np.cross(avg_n, mean_contact_frame[6:9])
            mean_contact_frame[9:] = mean_contact_frame[9:] / np.linalg.norm(mean_contact_frame[9:])

            contact_frames.append(mean_contact_frame.copy())
            contact_geom_idx.append(i + self.hand_geom_range[0])
            contact_ts_count.append(self.contact_num_count[i])
        return contact_frames, contact_geom_idx, contact_ts_count

    def compute_data_frame(self):
        self.motion_data.hand_kps.append(self.get_wbody_pos().reshape(-1, 3))
        self.motion_data.compensate_ft.append(np.concatenate([self.obj_vf.copy(), self.obj_vt.copy()]))
        self.motion_data.obj_pose.append(self.get_obj_qpos())
        contact_frames, contact_force, contact_body = self.get_contact()
        self.motion_data.contact_frames.append(contact_frames)
        self.motion_data.contact_force.append(contact_force)
        self.motion_data.contact_body.append(contact_body)
        self.motion_data.obj_vel.append(self.get_obj_qvel())
        self.motion_data.obj_acc.append(self.data.qacc[-6:].copy())

    def step(self, a):
        cfg = self.cc_cfg
        a = np.clip(a, -1, 1)

        self.prev_qpos = self.get_hand_qpos()
        self.prev_qvel = self.get_obj_qvel()
        # self.prev_bquat = self.bquat.copy()
        self.prev_obj_pos = self.get_obj_qpos()[:3]
        self.rfc_score = 0

        if self.cc_cfg.residual_force:
            self.obj_vf = self.cc_cfg.residual_force_scale * a[self.ndof:(self.ndof + 3)].copy()
            self.obj_vt = self.cc_cfg.residual_torque_scale * a[self.ndof + 3:self.ndof + 6].copy()
        else:
            self.obj_vf = self.obj_vt = np.zeros(3)

        fail = False
        # try:
        #     self.do_simulation(a, self.frame_skip)
        #     rf, rt, self.rfc_score = self.solve_rfc()
        #
        # except Exception as e:
        #     print("Exception in do_simulation", e, self.cur_t)
        #     fail = True
        self.do_simulation(a, self.frame_skip)
        rf, rt, self.rfc_score = self.solve_rfc()

        self.cur_t += 1

        # self.bquat = self.get_body_quat()
        # get obs
        # head_pos = self.get_wbody_pos(["Head"])
        reward = 1.0

        pos_diff, rot_diff, jpos_diff, obj_diff, obj_rot_diff = self.calc_ho_diff()
        body_fail = pos_diff > self.pos_diff_thresh
        body_fail = body_fail or rot_diff > self.rot_diff_thresh
        body_fail = body_fail or jpos_diff > self.jpos_diff_thresh
        body_fail = body_fail or obj_diff > self.obj_pos_diff_thresh
        body_fail = body_fail or obj_rot_diff > self.obj_rot_diff_thresh

        if self.mode == 'train':
            fail = fail or body_fail
        end = (self.cur_t + self.start_ind >= self.expert_len - self.w_size - 1)
        done = fail or end

        percent = self.cur_t / (self.expert_len - 1)
        obs = self.get_obs()
        return obs, reward, done, {"fail": fail, "end": end, "percent": percent}

    def calc_ho_diff(self):
        # Hand Diff
        expert_root_pose = self.get_expert_hand_root_pose()
        expert_jpos = self.get_expert_wbody_pos().reshape(-1, 3)

        cur_root_pose = self.get_hand_root_pose()
        cur_jpos = self.get_wbody_pos().reshape(-1, 3)

        pos_diff = np.linalg.norm(cur_root_pose[:3] - expert_root_pose[:3])
        jpos_diff = np.linalg.norm(cur_jpos - expert_jpos, axis=1).mean()

        quat_diff = quaternion_multiply(expert_root_pose[3:], quaternion_inverse(cur_root_pose[3:]))
        rot_diff = 2.0 * np.arcsin(np.clip(np.linalg.norm(quat_diff[1:]), 0, 1))

        # Obj Diff
        epose_obj = self.get_expert_obj_pose()
        obj_pose = self.get_obj_qpos()
        obj_diff = obj_pose[:3] - epose_obj[:3]
        obj_dist = np.linalg.norm(obj_diff)

        # object rotation reward
        obj_quat_diff = quaternion_multiply(epose_obj[3:], quaternion_inverse(epose_obj[3:]))
        obj_rot_diff = 2.0 * np.arcsin(np.clip(np.linalg.norm(obj_quat_diff[1:]), 0, 1))

        return pos_diff, rot_diff, jpos_diff, obj_dist, obj_rot_diff

    def reset_model(self, tracking=False):
        cfg = self.cc_cfg
        if not tracking:
            ind = 0
            self.start_ind = 0

            init_hand_pose_exp = self.expert["hand_dof_seq"][ind, :].copy()
            init_hand_vel_exp = self.expert["hand_dof_vel_seq"][ind, :].copy()  # Using GT joint velocity
            init_obj_pose_exp = self.expert["obj_pose_seq"][ind, :].copy()
            init_obj_vel_exp = self.expert["obj_vel_seq"][ind, :].copy()
            init_obj_ang_vel_exp = self.expert["obj_angle_vel_seq"][ind, :].copy()

            init_pose = np.concatenate([init_hand_pose_exp, init_obj_pose_exp])
            init_vel = np.concatenate([init_hand_vel_exp, init_obj_vel_exp, init_obj_ang_vel_exp])
            self.motion_data = MotionData()
        else:
            init_hand_pose_exp = self.get_expert_hand_qpos()
            init_hand_vel_exp = self.get_expert_hand_qvel()
            init_obj_pose_exp = self.get_expert_obj_pose()
            init_obj_vel_exp = self.get_expert_obj_vel()

            init_pose = np.concatenate([init_hand_pose_exp, init_obj_pose_exp])
            init_vel = np.concatenate([init_hand_vel_exp, init_obj_vel_exp])

        self.set_state(init_pose, init_vel)

        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_hand_qpos()[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_attr(self, attr, ind):
        ind = min(ind, self.expert_len - 1)
        return self.expert[attr][ind].copy()

    def get_expert_hand_qpos(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        expert_qpos = self.get_expert_attr("hand_dof_seq", ind)
        return expert_qpos

    def get_expert_hand_qvel(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        expert_vel = self.get_expert_attr("hand_dof_vel_seq", ind)
        return expert_vel

    def get_expert_hand_root_pose(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        expert_root_pos = self.get_expert_attr("body_pos_seq", ind)[0]
        expert_root_quat = self.get_expert_attr("body_quat_seq", ind)[0]
        return np.concatenate([expert_root_pos, expert_root_quat])

    def get_expert_obj_pose(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        obj_pose = self.get_expert_attr("obj_pose_seq", ind)
        return obj_pose

    def get_expert_obj_vel(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        obj_vel = self.get_expert_attr("obj_vel_seq", ind)
        obj_ang_vel = self.get_expert_attr("obj_angle_vel_seq", ind)
        return np.concatenate([obj_vel, obj_ang_vel])

    def get_expert_wbody_pos(self, delta_t=0):
        # world joint position
        ind = self.cur_t + delta_t + self.start_ind
        wbpos = self.get_expert_attr("body_pos_seq", ind)
        wbpos = wbpos.reshape(-1, 3).flatten()
        return wbpos

    def get_expert_wbody_quat(self, delta_t=0):
        # world joint position
        ind = self.cur_t + delta_t + self.start_ind
        wbquat = self.get_expert_attr("body_quat_seq", ind)
        wbquat = wbquat.reshape(-1, 4).flatten()
        return wbquat

    def get_expert_contact(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        contact_info = self.get_expert_attr("contact_info_seq", ind)
        return contact_info

    def get_expert_conf(self, delta_t=0):
        ind = self.cur_t + delta_t + self.start_ind
        conf = self.get_expert_attr("conf_seq", ind)
        return conf.copy()

    def get_hand_qpos(self):
        return self.data.qpos.copy()[:self.hand_qpos_dim]

    def get_hand_qvel(self):
        return self.data.qvel.copy()[:self.hand_qvel_dim]

    def get_obj_qpos(self):
        return self.data.qpos.copy()[self.hand_qpos_dim:]

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.hand_qvel_dim:]

    def get_wbody_quat(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.xquat[self.hand_body_idx].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.xquat[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_hand_root_pose(self):
        bone_id = mujoco.mj_name2id(self.model, 1, "link_palm")
        root_pos = self.data.xpos[bone_id]
        root_quat = self.data.xquat[bone_id]
        return np.concatenate((root_pos, root_quat))

    def get_wbody_pos(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.xpos[self.hand_body_idx].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.xpos[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            # ignore plane
            body_names = self.model.body_names[1:self.body_lim]
        else:
            body_names = selectList

        for body in body_names:
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)

        return np.concatenate(body_pos)

    def get_full_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            # ignore plane
            body_names = self.model.body_names[1:self.body_lim]
        else:
            body_names = selectList

        for body in body_names:
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)

        return np.concatenate(body_pos)

    def get_contact(self):
        contact_frame = []
        contact_body_idx = []
        contact_force = []
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            g1, g2 = contact.geom1, contact.geom2
            if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                    self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                contact_frame.append(np.concatenate([contact.pos.copy(), contact.frame.copy()]))
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force.append(c_array)
                contact_body_idx.append(g1)
        if len(contact_frame) == 0:
            return np.array([]), np.array([]), np.array([])
        return np.stack(contact_frame), np.stack(contact_force), np.stack(contact_body_idx)

    def record_contact(self):
        for i, contact in enumerate(self.data.contact):
            g1, g2 = contact.geom1, contact.geom2
            if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                    self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                self.contact_frame_arr[g1 - self.hand_geom_range[0]].append(np.concatenate([contact.pos, contact.frame]))
                self.contact_num_count[g1 - self.hand_geom_range[0]] += 1

    def get_contact_rwd(self):
        contact_info = {}
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                    self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                contact_info[g1] = contact.pos.copy()
        return contact_info

    def solve_rfc_torque(self, action):
        vf = action[self.ndof:(self.ndof + self.vf_dim)]
        v_torque = vf.copy()
        n_c = self.obj_contacts.shape[0]
        if n_c == 0:
            return v_torque

        J_arr = []
        for i in range(0, n_c):
            frame = self.obj_contacts[i, 3:12].reshape(3, 3)
            J_arr.append(frame[0])
        J = np.stack(J_arr).transpose()
        Q = 2 * cvxopt.matrix(np.matmul(J.T, J))
        p = cvxopt.matrix(-2 * np.matmul(J.T, v_torque))
        G = cvxopt.matrix(np.zeros((n_c, n_c)))
        h = cvxopt.matrix(np.zeros(n_c))
        sol = cvxopt.solvers.qp(Q, p, G, h, kktsolver='ldl', options={'kktreg': 1e-9, "show_progress": False})
        res = np.array(sol["x"]).ravel()
        rest_torque = v_torque - np.matmul(J, res)

        return rest_torque

    def get_target_ft(self):
        # Obj motion
        obj_quat = self.data.qpos[-4:]
        obj_mat = t3d.quaternions.quat2mat(obj_quat)
        obj_omega = self.data.geom_xvelr[-1]
        obj_a = self.data.qacc[-6:-3]
        obj_oa = self.data.qacc[-3:]

        # Obj inertia
        obj_mass = self.model.body_mass[-1]
        obj_inertia_b = np.diag(self.model.body_inertia[-1])
        obj_inertia = np.matmul(obj_mat, np.matmul(obj_inertia_b, np.transpose(obj_mat)))

        # Obj force and torque
        obj_force = obj_mass * (obj_a + np.array([0, 0, 9.8]))
        obj_torque = np.matmul(obj_inertia, obj_oa) + np.cross(obj_omega, np.matmul(obj_inertia, obj_omega))

        return obj_force, obj_torque

    def solve_rfc(self):
        # Solver Parameters
        dx = 0.0025
        mu = 1.0
        w_t = 1e4
        w_pn = 1.0      # 1.0
        w_pt = 1.0      # 1.0
        w_reg = 0

        contact_hand_geom_idx = self.avg_cp_geom
        contact_frame = self.avg_cps
        e_cps = []
        rel_vn_arr = []
        rel_vt_arr = []

        # Obj motion
        obj_p = self.data.qpos[-7:-4]
        obj_quat = self.data.qpos[-4:]
        obj_mat = t3d.quaternions.quat2mat(obj_quat)
        obj_v = self.geom_avg_vel[-1]
        obj_omega = self.geom_avg_ang_vel[-1]
        obj_a = self.obj_avg_acc[-6:-3]
        obj_oa = self.obj_avg_acc[-3:]

        # Obj inertia
        obj_mass = self.model.body_mass[-1]
        obj_inertia_b = np.diag(self.model.body_inertia[-1])
        obj_inertia = np.matmul(obj_mat, np.matmul(obj_inertia_b, np.transpose(obj_mat)))

        # Obj force and torque
        obj_force = obj_mass * (obj_a + np.array([0, 0, 9.8]))
        obj_torque = np.matmul(obj_inertia, obj_oa) + np.cross(obj_omega, np.matmul(obj_inertia, obj_omega))
        # obj_force = np.zeros(3)
        # obj_torque = self.obj_vt.copy()

        if len(contact_frame) == 0:
            return obj_force, obj_torque, np.linalg.norm(obj_force) + w_t * np.linalg.norm(obj_torque)

        # Expand Contact
        for i in range(len(contact_frame)):
            pos = contact_frame[i][:3]
            frame = contact_frame[i][3:12].reshape(3, 3)
            delta_pos = np.array([np.zeros(3), frame[1], -frame[1], frame[2], -frame[2]]) * dx
            g1 = contact_hand_geom_idx[i]

            for j in range(5):
                e_cps.append(np.concatenate([pos + delta_pos[j], contact_frame[i][3:12]], axis=0))
                # compute relative velocity at contact point
                cr_h = pos + delta_pos[j] - self.data.geom_xpos[g1]
                cr_o = pos + delta_pos[j] - obj_p
                cv_h = self.geom_avg_vel[g1] + np.cross(self.geom_avg_ang_vel[g1], cr_h)
                cv_o = obj_v + np.cross(obj_omega, cr_o)
                rel_v = cv_o - cv_h
                rel_vn = np.dot(frame[0], rel_v) * frame[0]
                rel_vt = rel_v - rel_vn
                rel_vn_arr.append(rel_vn)
                rel_vt_arr.append(rel_vt)

        n_c = len(e_cps)
        A_arr = []
        rA_arr = []
        p_n_arr = []
        p_t_arr = []
        for i in range(0, n_c):
            pos = e_cps[i][:3]
            frame = e_cps[i][3:12].reshape(3, 3)
            ts = self.cp_ts[i // 5]
            ts = ts / self.frame_skip

            # convert to polynomial friction cone
            x1 = (frame[0] + mu * frame[1]) / np.sqrt(1 + mu ** 2)
            x2 = (frame[0] - mu * frame[1]) / np.sqrt(1 + mu ** 2)
            x3 = (frame[0] + mu * frame[2]) / np.sqrt(1 + mu ** 2)
            x4 = (frame[0] - mu * frame[2]) / np.sqrt(1 + mu ** 2)
            A = np.stack([x1, x2, x3, x4]).T
            A_arr.append(ts * A)

            r = pos - obj_p
            r_x = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            rA_arr.append(ts * np.matmul(r_x, A))

            # No pressure when hand and object are seperating
            vn_dir = np.dot(rel_vn_arr[i], frame[0])
            if vn_dir <= 0:
                p_n_arr.append(np.linalg.norm(rel_vn_arr[i]) * np.ones(4))
            else:
                p_n_arr.append(np.zeros(4))

            # Friction force is along max dissipation direction when sliding
            dirs = np.array([np.dot(-rel_vt_arr[i], frame[1]),
                             np.dot(-rel_vt_arr[i], -frame[1]),
                             np.dot(-rel_vt_arr[i], frame[2]),
                             np.dot(-rel_vt_arr[i], -frame[2])])
            max_dis_idx = np.argmax(dirs)
            p_t = np.linalg.norm(rel_vt_arr[i]) * np.ones(4)
            p_t[max_dis_idx] = 0
            p_t_arr.append(p_t)

        J_f = np.concatenate(A_arr, axis=1)
        J_t = np.concatenate(rA_arr, axis=1)
        p_n = np.concatenate(p_n_arr, axis=0)
        p_t = np.concatenate(p_t_arr, axis=0)
        f_reg = np.ones(4 * n_c)

        # Solve QP
        # qpsolvers
        Q = 2 * (np.matmul(J_f.T, J_f) + w_t * np.matmul(J_t.T, J_t)) + 1e-7 * np.eye(4 * n_c)
        p = -2 * np.matmul(J_f.T, obj_force) - 2 * w_t * np.matmul(J_t.T, obj_torque) + \
            w_pn * p_n + w_pt * p_t + w_reg * f_reg
        G = -np.eye(4 * n_c)
        h = np.zeros(4 * n_c)
        res = qpsolvers.solve_qp(Q, p, G, h, solver='daqp')

        rest_force = obj_force - np.matmul(J_f, res)
        rest_torque = obj_torque - np.matmul(J_t, res)

        return rest_force, rest_torque, np.linalg.norm(rest_force) + math.sqrt(w_t) * np.linalg.norm(rest_torque)

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "hand":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        self.viewer_setup("rgb")
        return self.viewer

    def get_world_vf(self):
        return self.curr_vf

    def get_curr_torque(self):
        # Return current torque as list
        return self.curr_torque


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.render = True
    data_loader = DatasetGRAB(cfg.mujoco_model_file, cfg.data_specs)
    expert_seq = data_loader.load_seq(0, full_seq=True)
    env = HandObjMimic(cfg, expert_seq, data_loader.model_xml, cfg.data_specs, mode="test")
    env.reset()

    # eqpos = env.expert["hand_dof_seq"]
    # eqvel = env.expert["hand_dof_vel_seq"]
    # t = 0

    while True:
        action = env.action_space.sample()
        eqpos = env.get_expert_hand_qpos(delta_t=1)
        eqpos[2] += 0.2
        action[:env.ndof] = (eqpos - env.base_pose) / env.ctrl_scale
        action[env.ndof: env.ndof + env.vf_dim] = np.array([0, 0, 0])
        action[env.ndof + env.vf_dim:] = 0
        obs, reward, done, info = env.step(action)
        print(ho_mimic_reward(env, obs, action, info))
        if info['end']:
            env.reset()

        # env.data.qpos[:] = eqpos[t]
        # env.data.qvel[:] = eqvel[t]
        # env.sim.forward()
        # print(hand_mimic_reward(env, None, None, None))
        # env.cur_t = (env.cur_t + 1) % eqpos.shape[0]
        # t = (t + 1) % eqpos.shape[0]
