import os
import sys
import time

import numpy as np
import torch
import tqdm

sys.path.append(os.getcwd())

import pickle as pk
import mujoco
from uhc.data_loaders.mjxml.MujocoXML import MujocoXML
from uhc.utils.transforms import *
from uhc.khrylib.utils import *


# SingleDepth Dataset compatible with mujoco3
class DatasetSingleDepthNew:
    def __init__(self, model_fn, data_spec, noise=0.0, trans=(0, 0), mode='train'):
        """
        data_spec: dictionary with the following keys:
            "expert_fn": path to the expert motion file
            "motion_freq": frequency of the motion
            "sample_freq": frequency of the sampling
        """
        print("******* Reading Expert Data! ***********")
        expert_fn = data_spec["expert_fn"] if mode == 'train' else data_spec["test_expert_fn"]
        with open(expert_fn, "rb") as f:
            self.expert_data = pk.load(f)
        self.seq_name = data_spec["seq_name"]
        self.all_seqs = self.expert_data[self.seq_name]
        self.seq_num = len(self.all_seqs)
        self.single_clip_s = self.single_clip_e = None
        self.expert_seq = []
        self.noise = noise
        self.trans = trans
        self.max_len = data_spec.get('max_len', 600)
        self.has_gt = False

        self.hand_dof = self.all_seqs[0]['hand_pose_seq'].shape[-1]
        self.motion_freq = data_spec["motion_freq"]
        self.sample_freq = data_spec["sample_freq"]

        # load_model
        self.model_fn = model_fn
        self.with_obj = data_spec.get('with_obj', False)
        if self.with_obj:
            self.model_xml = self.add_obj(model_fn, data_spec['obj_fn'])
            self.model_path = 'dataset_model_temp.xml'
            with open(self.model_path, 'w') as f:
                f.write(self.model_xml)
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(self.model_fn)
        self.all_body_names = [mujoco.mj_id2name(self.model, 1, i) for i in range(self.model.nbody)]
        self.all_geom_names = [mujoco.mj_id2name(self.model, 5, i) for i in range(self.model.ngeom)]
        self.hand_body_names = [name for name in self.all_body_names if name.startswith("link")]
        self.hand_body_idx = [mujoco.mj_name2id(self.model, 1, name) for name in self.hand_body_names]
        self.hand_body_num = len(self.hand_body_names)    # all bodies except mujoco world and hand world
        self.joint_upper_limit = self.model.jnt_range[:, 1]
        self.joint_lower_limit = self.model.jnt_range[:, 0]
        if self.with_obj:
            self.joint_lower_limit = self.joint_lower_limit[:-1]
            self.joint_upper_limit = self.joint_upper_limit[:-1]

        for seq in self.all_seqs:
            self.preprocess_seq(seq)

        self.data_keys = [self.seq_name]
        self.curr_key = self.seq_name
        self.fr_start = 0

    def preprocess_seq(self, seq):
        hand_dof_seq = seq["hand_pose_seq"]
        raw_hand_dof_seq = hand_dof_seq.copy()
        obj_pose_seq = seq["obj_pose_seq"]
        seq_len = hand_dof_seq.shape[0]
        obj_init_pos = seq.get("obj_init_pos", None)
        conf_seq = seq.get("conf")

        gt_hand_dof_seq = seq.get('gt_hand_pose_seq')
        gt_obj_pose_seq = seq.get('gt_obj_pose_seq')
        if gt_hand_dof_seq is not None and gt_hand_dof_seq is not None:
            self.has_gt = True

        # clamp dof and add noise
        if self.noise > 0.0:
            noise_scale = self.joint_upper_limit - self.joint_lower_limit
            noise_scale[:6] = 0
            hand_dof_seq += 2 * self.noise * noise_scale * (np.random.random(size=hand_dof_seq.shape) - 0.5)
        hand_dof_seq[:, 0] += self.trans[0]
        hand_dof_seq[:, 1] += self.trans[1]
        obj_pose_seq[:, 0] += self.trans[0]
        obj_pose_seq[:, 1] += self.trans[1]
        hand_dof_seq = np.clip(hand_dof_seq, self.joint_lower_limit, self.joint_upper_limit)
        # raw_hand_dof_seq = np.clip(raw_hand_dof_seq, self.joint_lower_limit, self.joint_upper_limit)

        # compute extra information
        hand_dof_vel_seq, obj_vel_seq, obj_angle_vel_seq = self.compute_vel_from_seq(hand_dof_seq, obj_pose_seq)
        body_pos_seq, body_quat_seq = self.compute_body_pos_quat_from_seq(hand_dof_seq, seq_len)
        raw_body_pos_seq, raw_body_quat_seq = self.compute_body_pos_quat_from_seq(raw_hand_dof_seq, seq_len)
        contact_info = self.compute_contact_info(hand_dof_seq, obj_pose_seq, seq_len)
        gt_body_pos_seq, gt_body_quat_seq = None, None
        if self.has_gt:
            gt_body_pos_seq, gt_body_quat_seq = self.compute_body_pos_quat_from_seq(gt_hand_dof_seq, seq_len)

        if self.single_clip_s is not None:
            hand_dof_seq = hand_dof_seq[self.single_clip_s:self.single_clip_e]
            raw_hand_dof_seq = raw_hand_dof_seq[self.single_clip_s:self.single_clip_e]
            hand_dof_vel_seq = hand_dof_vel_seq[self.single_clip_s:self.single_clip_e]
            obj_pose_seq = obj_pose_seq[self.single_clip_s:self.single_clip_e]
            obj_vel_seq = obj_vel_seq[self.single_clip_s:self.single_clip_e]
            obj_angle_vel_seq = obj_angle_vel_seq[self.single_clip_s:self.single_clip_e]
            body_pos_seq = body_pos_seq[self.single_clip_s:self.single_clip_e]
            raw_body_pos_seq = raw_body_pos_seq[self.single_clip_s:self.single_clip_e]
            body_quat_seq = body_quat_seq[self.single_clip_s:self.single_clip_e]
            contact_info = contact_info[self.single_clip_s:self.single_clip_e]
            conf_seq = conf_seq[self.single_clip_s:self.single_clip_e]
            seq_len = self.single_clip_e - self.single_clip_s

        self.expert_seq.append({"hand_dof_seq": hand_dof_seq,
                                "raw_hand_dof_seq": raw_hand_dof_seq,
                                "hand_dof_vel_seq": hand_dof_vel_seq,
                                "obj_pose_seq": obj_pose_seq,
                                "obj_vel_seq": obj_vel_seq,
                                "obj_angle_vel_seq": obj_angle_vel_seq,
                                "body_pos_seq": body_pos_seq,
                                "raw_body_pos_seq": raw_body_pos_seq,
                                "body_quat_seq": body_quat_seq,
                                "contact_info_seq": contact_info,
                                "gt_hand_dof_seq": gt_hand_dof_seq,
                                "gt_obj_pose_seq": gt_obj_pose_seq,
                                "gt_body_pos_seq": gt_body_pos_seq,
                                "gt_body_quat_seq": gt_body_quat_seq,
                                "obj_init_pos": obj_init_pos,
                                "conf_seq": conf_seq,
                                "seq_len": seq_len})

    def add_obj(self, hand_fn, obj_fn):
        if obj_fn is None:
            raise ValueError("obj_fn is None!")
        mj_xml = MujocoXML(hand_fn)
        mj_xml.merge(MujocoXML(obj_fn))
        mj_xml.get_xml()
        return mj_xml.get_xml()

    def compute_vel_from_seq(self, hand_dof_seq, obj_pose_seq):
        """
        Compute hand dof velocity, object velocity and object angle velocity from the sequence
        """
        # compute hand dof velocity by finite difference
        hand_dof_vel_seq = np.zeros_like(hand_dof_seq)
        hand_dof_vel_seq[1:] = hand_dof_seq[1:] - hand_dof_seq[:-1]
        hand_dof_vel_seq[0] = hand_dof_vel_seq[1]

        # align rotation to [-pi, pi]
        rot_qpos_err = hand_dof_vel_seq[:, 3:6].copy()
        while np.any(rot_qpos_err > np.pi):
            rot_qpos_err[rot_qpos_err > np.pi] -= 2 * np.pi
        while np.any(rot_qpos_err < -np.pi):
            rot_qpos_err[rot_qpos_err < -np.pi] += 2 * np.pi
        hand_dof_vel_seq[:, 3:6] = rot_qpos_err[:]

        hand_dof_vel_seq = hand_dof_vel_seq * self.motion_freq

        # compute object velocity
        obj_vel_seq = np.zeros_like(obj_pose_seq[:, 0:3])
        obj_vel_seq[1:] = obj_pose_seq[1:, 0:3] - obj_pose_seq[:-1, 0:3]
        obj_vel_seq[0] = obj_vel_seq[1]
        obj_vel_seq = obj_vel_seq * self.motion_freq

        # compute object angle velocity
        obj_quat_seq = obj_pose_seq[:, 3:]
        obj_angle_vel_seq = np.zeros_like(obj_pose_seq[:, 0:3])
        rot_mat_seq = quaternion_to_matrix(torch.Tensor(obj_quat_seq))
        relative_rot_seq = torch.matmul(rot_mat_seq[1:], torch.transpose(rot_mat_seq[:-1], 1, 2))
        obj_angle_vel_seq[1:] = matrix_to_axis_angle(relative_rot_seq).numpy() * self.motion_freq
        obj_angle_vel_seq[0] = obj_angle_vel_seq[1]

        return hand_dof_vel_seq, obj_vel_seq, obj_angle_vel_seq

    def compute_contact_info(self, hand_dof_seq, obj_pose_seq, seq_len):
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

        sim_data = mujoco.MjData(self.model)
        contact_arr = []
        for i in range(seq_len):
            sim_data.qpos[:self.hand_dof] = hand_dof_seq[i]
            sim_data.qpos[self.hand_dof:] = obj_pose_seq[i]
            mujoco.mj_forward(self.model, sim_data)

            contact_info = {}
            # find contact btw hand and object
            for contact in sim_data.contact:
                g1, g2 = contact.geom1, contact.geom2
                if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                        self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                    contact_info[g1] = contact.pos.copy()
            contact_arr.append(contact_info)

        return np.stack(contact_arr)

    def compute_body_pos_quat_from_seq(self, hand_dof_seq, seq_len):
        """
        compute body position and quaternion from hand dof sequence
        :return: body_pos_seq: (seq_len, body_num, 3)
                 body_quat_seq: (seq_len, body_num, 4)
        """
        sim_data = mujoco.MjData(self.model)
        body_pos_seq = np.zeros((seq_len, self.hand_body_num, 3))
        body_quat_seq = np.zeros((seq_len, self.hand_body_num, 4))
        for i in range(seq_len):
            sim_data.qpos[:self.hand_dof] = hand_dof_seq[i]
            # sim.data.qpos[self.hand_dof:] = self.obj_pose_seq[i]
            mujoco.mj_forward(self.model, sim_data)
            body_pos_seq[i] = sim_data.xpos[self.hand_body_idx]
            body_quat_seq[i] = sim_data.xquat[self.hand_body_idx]
        return body_pos_seq, body_quat_seq

    def load_seq(self, seq_idx, start_idx=0, full_seq=False, end_idx=-1):
        """
        load sequence from the expert data
        """
        hand_dof_seq = self.expert_seq[seq_idx]["hand_dof_seq"]
        raw_hand_dof_seq = self.expert_seq[seq_idx]["raw_hand_dof_seq"]
        hand_dof_vel_seq = self.expert_seq[seq_idx]["hand_dof_vel_seq"]
        obj_pose_seq = self.expert_seq[seq_idx]["obj_pose_seq"]
        obj_vel_seq = self.expert_seq[seq_idx]["obj_vel_seq"]
        obj_angle_vel_seq = self.expert_seq[seq_idx]["obj_angle_vel_seq"]
        body_pos_seq = self.expert_seq[seq_idx]["body_pos_seq"]
        raw_body_pos_seq = self.expert_seq[seq_idx]["raw_body_pos_seq"]
        body_quat_seq = self.expert_seq[seq_idx]["body_quat_seq"]
        contact_info_seq = self.expert_seq[seq_idx]["contact_info_seq"]
        conf_seq = self.expert_seq[seq_idx]["conf_seq"]
        seq_len = self.expert_seq[seq_idx]["seq_len"]

        clip_s = min(start_idx, seq_len)
        clip_e = min(clip_s + self.max_len, seq_len)

        gt_hand_dof_seq, gt_obj_pose_seq, gt_body_pos_seq, gt_body_quat_seq = None, None, None, None
        if self.has_gt:
            gt_hand_dof_seq = self.expert_seq[seq_idx]["gt_hand_dof_seq"][np.arange(clip_s, clip_e, self.sample_freq)]
            gt_obj_pose_seq = self.expert_seq[seq_idx]["gt_obj_pose_seq"][np.arange(clip_s, clip_e, self.sample_freq)]
            gt_body_pos_seq = self.expert_seq[seq_idx]["gt_body_pos_seq"][np.arange(clip_s, clip_e, self.sample_freq)]
            gt_body_quat_seq = self.expert_seq[seq_idx]["gt_body_quat_seq"][np.arange(clip_s, clip_e, self.sample_freq)]

        if full_seq:
            clip_e = seq_len

        if end_idx > 0:
            clip_e = end_idx

        return {
            "hand_dof_seq": hand_dof_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "raw_hand_dof_seq": raw_hand_dof_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "hand_dof_vel_seq": hand_dof_vel_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "obj_pose_seq": obj_pose_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "obj_vel_seq": obj_vel_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "obj_angle_vel_seq": obj_angle_vel_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "body_pos_seq": body_pos_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "raw_body_pos_seq": raw_body_pos_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "body_quat_seq": body_quat_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "contact_info_seq": contact_info_seq[np.arange(clip_s, clip_e, self.sample_freq)],
            "gt_hand_dof_seq": gt_hand_dof_seq,
            "gt_obj_pose_seq": gt_obj_pose_seq,
            "gt_body_pos_seq": gt_body_pos_seq,
            "gt_body_quat_seq": gt_body_quat_seq,
            "conf_seq": conf_seq[np.arange(clip_s, clip_e, self.sample_freq)] if conf_seq is not None else None,
        }

    def load_frame(self, seq_idx, ind):
        return {
            "hand_dof_seq": self.expert_seq[seq_idx]["hand_dof_seq"][ind],
            "raw_hand_dof_seq": self.expert_seq[seq_idx]["raw_hand_dof_seq"][ind],
            "hand_dof_vel_seq": self.expert_seq[seq_idx]["hand_dof_vel_seq"][ind],
            "obj_pose_seq": self.expert_seq[seq_idx]["obj_pose_seq"][ind],
            "obj_vel_seq": self.expert_seq[seq_idx]["obj_vel_seq"][ind],
            "obj_angle_vel_seq": self.expert_seq[seq_idx]["obj_angle_vel_seq"][ind],
            "body_pos_seq": self.expert_seq[seq_idx]["body_pos_seq"][ind],
            "raw_body_pos_seq": self.expert_seq[seq_idx]["raw_body_pos_seq"][ind],
            "body_quat_seq": self.expert_seq[seq_idx]["body_quat_seq"][ind],
            "contact_info_seq": self.expert_seq[seq_idx]["contact_info_seq"][ind],
            "conf_seq": self.expert_seq[seq_idx]["conf_seq"][ind]
        }

    def get_len(self, seq_idx):
        return self.expert_seq[seq_idx]["seq_len"]

    def get_obj_init_pos(self, seq_idx):
        init_pose = self.expert_seq[seq_idx].get("obj_init_pos", None)
        if init_pose is None:
            init_pose = torch.zeros(3)
        return init_pose

    # def show_expert(self):
    #     """
    #     Visualize the expert motion
    #     """
    #     sim = mjpy.MjSim(self.model)
    #     viewer = mjpy.MjViewer(sim)
    #     for i in tqdm.tqdm(range(len(self.expert_seq))):
    #         seq = self.expert_seq[i]
    #         for t in range(len(seq["hand_dof_seq"])):
    #             sim.data.qpos[:self.hand_dof] = seq["hand_dof_seq"][t]
    #             sim.data.qpos[self.hand_dof:] = seq["obj_pose_seq"][t]
    #             sim.forward()
    #             viewer.render()
    #             time.sleep(0.01)


if __name__ == '__main__':
    model_fn = 'assets/hand_model/manohand_reduce2/mano_hand.xml'
    data_spec = {
        "dataset_name": "Tracking",
        "seq_name": "box_seq",
        "expert_fn": "sample_data/SingleDepth/box_seq.pkl",
        "clip_s": 0,
        "clip_e": 150,
        "max_len": 800,
        "motion_freq": 30,
        "sample_freq": 1,
        "with_obj": True,
        "obj_fn": "assets/SingleDepth/box_light.xml",
    }
    dataset = DatasetSingleDepthNew(model_fn, data_spec)
    dataset.show_expert()
