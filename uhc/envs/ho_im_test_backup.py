import argparse
import os
import sys

import cvxopt
import mujoco_py
import transforms3d.quaternions
from gym import spaces
from mujoco_py import functions as mjf
from scipy.linalg import cho_solve, cho_factor

from uhc.data_loaders.dataset_grab import DatasetGRAB
from uhc.utils.config_utils.handmimic_config import Config

sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from uhc.khrylib.rl.envs.common import mujoco_env

from uhc.utils.transformation import (
    quat_mul_vec,
)
from uhc.utils.math_utils import *
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
)


class HandObjMimicTest(mujoco_env.MujocoEnv):
    def __init__(self, cfg, init_expert_seq, model_path, data_specs, mode="train"):
        self.sim_step = cfg.sim_step
        mujoco_env.MujocoEnv.__init__(self, model_path, self.sim_step)

        self.w_size = cfg.future_w_size
        self.expert_window = init_expert_seq
        self.expert_index = [i for i in range(self.w_size + 1)]

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

        # load hand model
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()

        # read some params from model
        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.hand_qpos_dim = self.model.nq - 7
        self.hand_qvel_dim = self.sim.model.nv - 6
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.hand_body_names = [name for name in self.model.body_names if name.startswith("link")]
        self.hand_body_idx = [self.model.body_name2id(name) for name in self.hand_body_names]
        self.obj_body_idx = self.hand_body_idx[-1] + 1
        self.obj_body_idx = self.hand_body_idx[-1] + 1
        self.hand_geom_range = [-1, -1]
        for i, name in enumerate(self.model.geom_names):
            if name.startswith('robot0:') and self.hand_geom_range[0] == -1:
                self.hand_geom_range[0] = i
            if (not name.startswith('robot0:')) and self.hand_geom_range[0] != -1:
                self.hand_geom_range[1] = i - 1
                break
        self.obj_geom_idx = self.hand_geom_range[1] + 1
        print("Hand geom range: ", self.hand_geom_range)
        print("Object geom idx: ", self.obj_geom_idx)
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
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)

    def insert_new_frame(self, frame):
        expire_idx = self.expert_index[0]
        self.expert_window[expire_idx] = frame
        for i in range(self.w_size + 1):
            self.expert_index[i] = (self.expert_index[i] + 1) % (self.w_size + 1)

    def set_mode(self, mode):
        self.mode = mode

    def set_action_spaces(self):
        cfg = self.cc_cfg
        self.vf_dim = 0
        self.meta_pd_dim = 0
        body_id_list = self.model.geom_bodyid.tolist()

        if cfg.residual_force:
            self.vf_dim = 3

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

    def get_obs(self):
        if self.obs_type == 0:
            obs = self.get_full_obs()
        elif self.obs_type == 1:
            obs = self.get_full_obs_v2()
        elif self.obs_type == 2:
            obs = self.get_full_obs_v3()
        else:
            obs = self.get_full_obs_v4(self.w_size)
        return obs

    def get_full_obs(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        obs = []

        ######## target hand pose #########
        target_hand_qpos = self.get_expert_hand_qpos(delta_t=1 + delta_t)

        obs.append(target_hand_qpos[:])
        obs.append(qpos[:self.hand_qpos_dim])

        ################ hand vels ################
        # vel
        obs.append(qvel[:self.hand_qvel_dim])

        ######### relative root position and orientation #########
        target_root_pose = self.get_expert_hand_root_pose(delta_t=1 + delta_t)
        cur_root_pose = self.get_hand_root_pose()
        rel_root_pos = target_root_pose[:3] - cur_root_pose[:3]
        rel_root_rot = quaternion_multiply(target_root_pose[3:], quaternion_inverse(cur_root_pose[3:]))
        obs.append(rel_root_pos)
        obs.append(rel_root_rot)

        ############# target/difference joint positions ############
        target_jpos = self.get_expert_wbody_pos(delta_t=1 + delta_t).reshape(-1, 3)
        curr_jpos = self.get_wbody_pos().reshape(-1, 3)

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, cur_root_pose[3:])  # body frame position
        obs.append(r_jpos.ravel())
        diff_jpos = target_jpos - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, cur_root_pose[3:])
        obs.append(diff_jpos.ravel())

        ############# object pose  ############
        obs.append(qpos[self.hand_qpos_dim:])

        ############# object vel  ############
        obs.append(qvel[self.hand_qvel_dim:])

        ############# object target pose relative to current pose ############
        target_obj_pose = self.get_expert_obj_pose(delta_t=1 + delta_t)
        cur_obj_pose = self.get_obj_qpos()
        rel_obj_pos = target_obj_pose[:3] - cur_obj_pose[:3]
        rel_obj_rot = quaternion_multiply(target_obj_pose[3:], quaternion_inverse(cur_obj_pose[3:]))
        obs.append(rel_obj_pos)
        obs.append(rel_obj_rot)

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v2(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        obs = []

        cur_root_pose = self.get_hand_root_pose()

        ######## target hand pose #########
        target_hand_qpos = self.get_expert_hand_qpos(delta_t=1 + delta_t)

        obs.append(target_hand_qpos[3:])
        obs.append(qpos[3:self.hand_qpos_dim])

        ################ hand vels ################
        # vel
        obs.append(qvel[6:self.hand_qvel_dim])
        # transform global vel to local coordinates
        local_root_vel = transform_vec(qvel[:3], cur_root_pose[3:])
        local_root_ang_vel = transform_vec(qvel[3:6], cur_root_pose[3:])
        obs.append(local_root_vel)
        obs.append(local_root_ang_vel)

        ######### relative root position and orientation #########
        target_root_pose = self.get_expert_hand_root_pose(delta_t=1 + delta_t)
        rel_root_pos = target_root_pose[:3] - cur_root_pose[:3]
        rel_root_pos = transform_vec(rel_root_pos, cur_root_pose[3:])
        rel_root_rot = quaternion_multiply(target_root_pose[3:], quaternion_inverse(cur_root_pose[3:]))
        obs.append(rel_root_pos)
        obs.append(rel_root_rot)

        ############# target/difference joint positions ############
        target_jpos = self.get_expert_wbody_pos(delta_t=1 + delta_t).reshape(-1, 3)
        curr_jpos = self.get_wbody_pos().reshape(-1, 3)

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, cur_root_pose[3:])  # body frame position
        obs.append(r_jpos.ravel())
        diff_jpos = target_jpos - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, cur_root_pose[3:])
        obs.append(diff_jpos.ravel())

        ############# object pose  ############
        obj_root_pose = qpos[self.hand_qpos_dim:]
        obj_pos_rel_hand = obj_root_pose[:3] - cur_root_pose[:3]
        obj_pos_rel_hand = transform_vec(obj_pos_rel_hand, cur_root_pose[3:])
        obj_rot_rel_hand = quaternion_multiply(obj_root_pose[3:], quaternion_inverse(cur_root_pose[3:]))
        obs.append(obj_pos_rel_hand)
        obs.append(obj_rot_rel_hand)

        ############# object vel  ############
        obj_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim:self.hand_qvel_dim + 3], cur_root_pose[3:])
        obj_ang_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim + 3:self.hand_qvel_dim + 6], cur_root_pose[3:])
        obs.append(obj_vel_rel_hand)
        obs.append(obj_ang_vel_rel_hand)

        ############# object target pose relative to current pose ############
        target_obj_pose = self.get_expert_obj_pose(delta_t=1 + delta_t)
        cur_obj_pose = self.get_obj_qpos()
        rel_obj_pos = target_obj_pose[:3] - cur_obj_pose[:3]
        rel_obj_pos = transform_vec(rel_obj_pos, cur_root_pose[3:])
        rel_obj_rot = quaternion_multiply(target_obj_pose[3:], quaternion_inverse(cur_obj_pose[3:]))
        rel_obj_rot = quaternion_multiply(rel_obj_rot, quaternion_inverse(cur_root_pose[3:]))
        obs.append(rel_obj_pos)
        obs.append(rel_obj_rot)

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v3(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        cur_root_pose = self.get_hand_root_pose()
        obs = []

        ######## current hand pose #########
        obs.append(cur_root_pose[3:])
        obs.append(qpos[6:self.hand_qpos_dim])

        ######## target hand pose relative to current pose #########
        target_hand_qpos = self.get_expert_hand_qpos(delta_t=1 + delta_t)
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
        target_root_pose = self.get_expert_hand_root_pose(delta_t=1 + delta_t)
        rel_root_pos = target_root_pose[:3] - cur_root_pose[:3]
        rel_root_pos = transform_vec(rel_root_pos, cur_root_pose[3:])
        rel_root_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), target_root_pose[3:])
        obs.append(rel_root_pos)
        obs.append(rel_root_rot)

        ############# target/difference joint positions ############
        target_jpos = self.get_expert_wbody_pos(delta_t=1 + delta_t).reshape(-1, 3)
        curr_jpos = self.get_wbody_pos().reshape(-1, 3)

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos[1:] - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, cur_root_pose[3:])  # body frame position
        obs.append(r_jpos.ravel())
        diff_jpos = target_jpos[1:] - curr_jpos[1:]
        diff_jpos = transform_vec_batch(diff_jpos, cur_root_pose[3:])
        obs.append(diff_jpos.ravel())

        ############# object pose  ############
        obj_root_pose = qpos[self.hand_qpos_dim:]
        obj_pos_rel_hand = obj_root_pose[:3] - cur_root_pose[:3]
        obj_pos_rel_hand = transform_vec(obj_pos_rel_hand, cur_root_pose[3:])
        obj_rot_rel_hand = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), obj_root_pose[3:])
        obs.append(obj_pos_rel_hand)
        obs.append(obj_rot_rel_hand)

        ############# object vel  ############
        obj_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim:self.hand_qvel_dim + 3], cur_root_pose[3:])
        obj_ang_vel_rel_hand = transform_vec(qvel[self.hand_qvel_dim + 3:self.hand_qvel_dim + 6], cur_root_pose[3:])
        obs.append(obj_vel_rel_hand)
        obs.append(obj_ang_vel_rel_hand)

        ############# object target pose relative to current pose ############
        target_obj_pose = self.get_expert_obj_pose(delta_t=1 + delta_t)
        cur_obj_pose = obj_root_pose.copy()
        rel_obj_pos = target_obj_pose[:3] - cur_obj_pose[:3]
        rel_obj_pos = transform_vec(rel_obj_pos, cur_root_pose[3:])
        rel_obj_rot = quaternion_multiply(target_obj_pose[3:], quaternion_inverse(cur_obj_pose[3:]))
        rel_obj_rot = quaternion_multiply(quaternion_inverse(cur_root_pose[3:]), rel_obj_rot)
        obs.append(rel_obj_pos)
        obs.append(rel_obj_rot)

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v4(self, w_size=1):
        data = self.data
        qpos = data.qpos[:self.qpos_dim].copy()
        qvel = data.qvel[:self.qvel_dim].copy()
        cur_root_pose = self.get_hand_root_pose()
        obs = []

        ######## current hand pose #########
        obs.append(cur_root_pose[3:])
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
            obs.append(rel_root_pos)
            obs.append(rel_root_rot)

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
        obs.append(obj_pos_rel_hand)
        obs.append(obj_rot_rel_hand)

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
            obs.append(rel_obj_pos)
            obs.append(rel_obj_rot)

        obs = np.concatenate(obs)
        return obs

    def fail_safe(self):
        self.data.qpos[:self.qpos_dim] = self.get_expert_qpos()
        self.data.qvel[:self.qvel_dim] = self.get_expert_qvel()
        self.sim.forward()

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv

        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(nv, nv)
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
        target_pos = np.zeros_like(qpos)

        ctrl_pos = ctrl[:3]
        ctrl_rot = ctrl[3:7] if cfg.grot_type == 'quat' else ctrl[3:6]
        ctrl_loco = ctrl[7:self.ndof + 1] if cfg.grot_type == 'quat' else ctrl[6:self.ndof]

        if cfg.pd_type == 'base':
            target_pos[0:3] = self.base_pose[0:3] + self.ctrl_scale[0:3] * ctrl_pos
            target_pos[6:] = self.base_pose[6:] + self.ctrl_scale[6:] * ctrl_loco
        else:
            target_pos[0:3] = qpos[0:3] + self.ctrl_scale[0:3] * ctrl_pos
            target_pos[6:] = qpos[6:] + self.ctrl_scale[6:] * ctrl_loco

        if self.cc_cfg.grot_type == 'euler':
            target_pos[3:6] = (self.base_pose[3:6] if cfg.pd_type == 'base' else qpos[3:6]) \
                              + self.ctrl_scale[3:6] * ctrl_rot
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

    def rfc_explicit(self, vf):
        qfrc = np.zeros(self.data.qfrc_applied.shape)
        num_each_body = self.cc_cfg.get("residual_force_bodies_num", 1)
        residual_contact_only = self.cc_cfg.get("residual_contact_only", False)
        residual_contact_only_ground = self.cc_cfg.get("residual_contact_only_ground", False)
        residual_contact_projection = self.cc_cfg.get("residual_contact_projection", False)
        vf_return = np.zeros(vf.shape)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            foot_pos = self.data.get_body_xpos(body)[2]
            has_contact = False

            geom_id = self.vf_geoms[i]
            for contact in self.data.contact[:self.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if (g1 == 0 and g2 == geom_id) or (g2 == 0 and g1 == geom_id):
                    has_contact = True
                    break

            if residual_contact_only_ground:
                pass
            else:
                has_contact = foot_pos <= 0.12

            if not (residual_contact_only and not has_contact):
                for idx in range(num_each_body):
                    contact_point = vf[(i * num_each_body + idx) * self.body_vf_dim:(
                                                                                            i * num_each_body + idx) * self.body_vf_dim + 3]
                    if residual_contact_projection:
                        contact_point = self.smpl_robot.project_to_body(body, contact_point)

                    force = (vf[(i * num_each_body + idx) * self.body_vf_dim + 3:(
                                                                                         i * num_each_body + idx) * self.body_vf_dim + 6] * self.cc_cfg.residual_force_scale)
                    torque = (vf[(i * num_each_body + idx) * self.body_vf_dim + 6:(
                                                                                          i * num_each_body + idx) * self.body_vf_dim + 9] * self.cc_cfg.residual_force_scale if self.cc_cfg.residual_force_torque else np.zeros(
                        3))

                    contact_point = self.pos_body2world(body, contact_point)

                    force = self.vec_body2world(body, force)
                    torque = self.vec_body2world(body, torque)

                    vf_return[(i * num_each_body + idx) * self.body_vf_dim:(
                                                                                   i * num_each_body + idx) * self.body_vf_dim + 3] = contact_point
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim + 3:(
                                                                                       i * num_each_body + idx) * self.body_vf_dim + 6] = (
                            force / self.cc_cfg.residual_force_scale)

                    # print(np.linalg.norm(force), np.linalg.norm(torque))
                    mjf.mj_applyFT(
                        self.model,
                        self.data,
                        force,
                        torque,
                        contact_point,
                        body_id,
                        qfrc,
                    )
        self.curr_vf = vf_return
        self.data.qfrc_applied[:] = qfrc

    def rfc_implicit(self, vf):
        vf *= self.cc_cfg.residual_force_scale * self.rfc_rate
        curr_root_quat = self.remove_base_rot(self.get_humanoid_qpos()[3:7])
        hq = get_heading_q(curr_root_quat)
        # hq = get_heading_q(self.get_humanoid_qpos()[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        vf = np.clip(vf, -self.cc_cfg.residual_force_lim, self.cc_cfg.residual_force_lim)
        self.data.qfrc_applied[:vf.shape[0]] = vf

    def rfc_obj(self, vf):
        vf = vf.astype(np.float64)
        obj_mc = self.get_obj_qpos()[:3]
        qfrc = np.zeros(self.data.qfrc_applied.shape).astype(np.float64)

        mjf.mj_applyFT(
            self.model,
            self.data,
            np.zeros(3).astype(np.float64),
            vf[:],
            obj_mc,
            self.obj_body_idx,
            qfrc,
        )
        self.data.qfrc_applied[:] = qfrc

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl = action
        self.curr_torque = []
        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl, i_iter=i)
            elif cfg.action_type == "torque":
                torque = ctrl * self.a_scale * 100
            torque = np.clip(torque, -self.torque_lim, self.torque_lim)

            self.curr_torque.append(torque)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[self.ndof:(self.ndof + self.vf_dim)].copy()
                self.rfc_obj(vf)
            self.sim.step()
            if self.viewer:
                self.viewer.render()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):
        cfg = self.cc_cfg
        a = np.clip(a, -1, 1)

        self.prev_qpos = self.get_hand_qpos()
        self.prev_qvel = self.get_obj_qvel()
        # self.prev_bquat = self.bquat.copy()
        self.prev_obj_pos = self.get_obj_qpos()[:3]
        self.obj_contacts = self.get_contact()

        fail = False
        try:
            self.do_simulation(a, self.frame_skip)
        except Exception as e:
            print("Exception in do_simulation", e, self.cur_t)
            fail = True
        # self.do_simulation(a, self.frame_skip)

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
        done = fail

        obs = self.get_obs()
        return obs, reward, done, {"fail": fail}

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

    def reset_model(self):
        cfg = self.cc_cfg

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

    def get_hand_qpos(self):
        return self.data.qpos.copy()[:self.hand_qpos_dim]

    def get_hand_qvel(self):
        return self.data.qvel.copy()[:self.hand_qvel_dim]

    def get_obj_qpos(self):
        return self.data.qpos.copy()[self.hand_qpos_dim:]

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.hand_qvel_dim:]

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = SMPL_EE_NAMES
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_wbody_quat(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xquat[self.hand_body_idx].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xquat[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_hand_root_pose(self):
        bone_id = self.model._body_name2id["link_palm"]
        root_pos = self.data.body_xpos[bone_id]
        root_quat = self.data.body_xquat[bone_id]
        return np.concatenate((root_pos, root_quat))

    def get_wbody_pos(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xpos[self.hand_body_idx].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xpos[bone_idx]
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
        obj_contact = []
        for contact in self.data.contact[:self.data.ncon]:
            g1, g2 = contact.geom1, contact.geom2
            if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and g2 == self.obj_geom_idx:
                obj_contact.append(np.concatenate([contact.pos.copy(), contact.frame.copy()]))
        if len(obj_contact) == 0:
            return np.array([])
        return np.stack(obj_contact)

    def solve_rfc(self, action):
        vf = action[self.ndof:(self.ndof + self.vf_dim)]
        v_force = vf[:3].copy() * self.cc_cfg.residual_force_scale
        v_torque = vf[3:6].copy()
        n_c = self.obj_contacts.shape[0]
        if n_c == 0:
            return vf

        friction_coef = 1.0
        obj_c = self.prev_obj_pos.copy()
        A_arr = []
        rA_arr = []
        for i in range(0, n_c):
            pos = self.obj_contacts[i, :3]
            frame = self.obj_contacts[i, 3:12].reshape(3, 3)

            # convert to polynomial friction cone
            x1 = (frame[0] + friction_coef * frame[1]) / np.sqrt(1 + friction_coef ** 2)
            x2 = (frame[0] - friction_coef * frame[1]) / np.sqrt(1 + friction_coef ** 2)
            x3 = (frame[0] + friction_coef * frame[2]) / np.sqrt(1 + friction_coef ** 2)
            x4 = (frame[0] - friction_coef * frame[2]) / np.sqrt(1 + friction_coef ** 2)
            A = np.stack([x1, x2, x3, x4]).T
            A_arr.append(A)

            r = pos - obj_c
            r_x = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            rA_arr.append(np.matmul(r_x, A))

        J_f = np.concatenate(A_arr, axis=1)
        J_t = np.concatenate(rA_arr, axis=1)

        # Solve QP
        w_t = self.cc_cfg.residual_force_scale ** 2
        Q = 2 * cvxopt.matrix(np.matmul(J_f.T, J_f) + w_t * np.matmul(J_t.T, J_t))
        p = cvxopt.matrix(-2 * np.matmul(J_f.T, v_force) - 2 * w_t * np.matmul(J_t.T, v_torque))
        G = -cvxopt.matrix(np.eye(4 * n_c))
        h = cvxopt.matrix(np.zeros(4 * n_c))
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(Q, p, G, h)
        res = np.array(sol["x"]).ravel()

        rest_force = v_force - np.matmul(J_f, res)
        rest_torque = v_torque - np.matmul(J_t, res)

        ret = np.concatenate([rest_force / self.cc_cfg.residual_force_scale, rest_torque])

        return ret

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


def ho_mimic_reward_test(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.5)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 100)
    )

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_hand_qpos()
    e_qvel = env.get_expert_hand_qvel()
    e_wbquat = env.get_expert_wbody_quat().reshape(-1, 4)
    e_wbpos = env.get_expert_wbody_pos().reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = (pose_diff ** 2).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = (wpose_diff ** 2).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity rewards
    vel_dist = cur_qvel - e_qvel
    vel_dist = (vel_dist ** 2).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_obj_pose()
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.power(np.linalg.norm(obj_diff), 2)
    obj_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = obj_rot_diff[0] ** 2
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = env.get_expert_obj_vel()
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = (obj_vel_diff ** 2).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    # np.set_printoptions(precision=5, suppress=1)
    # print(np.array([reward, vel_dist, pose_diff, wpose_diff]), \
    # np.array([pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]))
    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_reward, obj_rot_reward, obj_vel_reward, obj_rfc_reward]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.render = True
    data_loader = DatasetGRAB(cfg.mujoco_model_file, cfg.data_specs)
    expert_seq = data_loader.load_seq(3)
    env = HandObjMimicTest(cfg, expert_seq, data_loader.model_xml, cfg.data_specs, mode="test")
    env.reset()

    # eqpos = env.expert["hand_dof_seq"]
    # eqvel = env.expert["hand_dof_vel_seq"]
    # t = 0

    while True:
        action = env.action_space.sample()
        eqpos = env.get_expert_hand_qpos(delta_t=1)
        action[:env.ndof] = 0
        action[env.ndof: env.ndof + env.vf_dim] = np.array([0, 0, 0, 0, 0, 0])
        action[env.ndof + env.vf_dim:] = 0
        obs, reward, done, info = env.step(action)
        # print(ho_mimic_reward(env, obs, action, info))
        if info['end']:
            env.reset()

        # env.data.qpos[:] = eqpos[t]
        # env.data.qvel[:] = eqvel[t]
        # env.sim.forward()
        # print(hand_mimic_reward(env, None, None, None))
        # env.cur_t = (env.cur_t + 1) % eqpos.shape[0]
        # t = (t + 1) % eqpos.shape[0]
