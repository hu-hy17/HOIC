import argparse
import os
import sys
import multiprocessing
import queue
import time

import mujoco_py as mjpy
from uhc.envs.ho_im_test import HandObjMimicTest
from uhc.khrylib.rl.core import PolicyGaussian
from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.core.critic import Value
from uhc.utils.tools import CustomUnpickler
from uhc.utils.transforms import quaternion_to_matrix, matrix_to_axis_angle
import torch
import numpy as np

from uhc.utils.config_utils.handmimic_config import Config
from uhc.data_loaders.mjxml.MujocoXML import MujocoXML

frame_queue = multiprocessing.Queue()
is_show_contact = True


def load_display_xml(cfg):
    hand_model_file = cfg.vis_model_file
    obj_model_file = cfg.data_specs['obj_fn']
    hand_model = MujocoXML(hand_model_file)
    obj_model = MujocoXML(obj_model_file)
    ref_obj_model = MujocoXML(obj_model_file)
    # ref_obj_model2 = MujocoXML(obj_model_file)

    ref_obj_model.worldbody.getchildren()[0].attrib['name'] += '_ref'
    for ele in ref_obj_model.worldbody.getchildren()[0].getchildren():
        if ele.tag == 'geom':
            ele.attrib['material'] = "object_ref"
            ele.attrib['name'] += "_ref"
            # ele.attrib['group'] = "2"

    hand_model.merge(obj_model)
    hand_model.merge(ref_obj_model)
    # hand_model.merge(ref_obj_model2)
    return hand_model.get_xml()


def render_worker(frame_q: multiprocessing.Queue, cfg):
    # setup display model
    display_model_path = 'display_model_tmp.xml'
    display_xml = load_display_xml(cfg)
    with open(display_model_path, 'w') as f:
        f.write(display_xml)
    display_model = mjpy.load_model_from_path(display_model_path)
    display_sim = mjpy.MjSim(display_model)
    display_sim_back = mjpy.MjSim(display_model)
    hand_qpos_dim = 26

    # prepare for contact render
    hand_geom_range = [-1, -1]
    ref_hand_geom_range = [-1, -1]
    for i, name in enumerate(display_model.geom_names):
        if name.startswith('robot0:') and hand_geom_range[0] == -1:
            hand_geom_range[0] = i
        if (not name.startswith('robot0:')) and hand_geom_range[0] != -1:
            hand_geom_range[1] = i - 1
            break
    for i, name in enumerate(display_model.geom_names):
        if name.startswith('robot1:') and ref_hand_geom_range[0] == -1:
            ref_hand_geom_range[0] = i
        if (not name.startswith('robot1:')) and ref_hand_geom_range[0] != -1:
            ref_hand_geom_range[1] = i - 1
            break

    obj_geom_range = [-1, -1]
    ref_obj_geom_range = [-1, -1]
    for i in range(ref_hand_geom_range[1] + 1, len(display_model.geom_names)):
        name = display_model.geom_names[i]
        if name.startswith('C_') and obj_geom_range[0] == -1:
            obj_geom_range[0] = i
        if (not name.startswith('C_')) and obj_geom_range[0] != -1:
            obj_geom_range[1] = i - 1
            break

    for i in range(obj_geom_range[1] + 1, len(display_model.geom_names)):
        name = display_model.geom_names[i]
        if name.startswith('C_') and ref_obj_geom_range[0] == -1:
            ref_obj_geom_range[0] = i
        if (not name.startswith('C_')) and ref_obj_geom_range[0] != -1:
            ref_obj_geom_range[1] = i - 1
            break

    viewer = mjpy.MjViewer(display_sim)
    viewer2 = mjpy.MjViewer(display_sim_back)

    while True:
        if frame_q.empty():
            viewer.render()
            viewer2.render()
            continue
        print("Update Viewer")
        frame = frame_q.get()
        frame['hand_pose'][0] -= 0.1
        frame['obj_pose'][0] -= 0.1
        frame['e_hand_pose'][0] += 0.1
        frame['e_obj_pose'][0] += 0.1
        display_sim.data.qpos[:hand_qpos_dim] = frame['hand_pose']
        display_sim.data.qpos[hand_qpos_dim: 2 * hand_qpos_dim] = frame['e_hand_pose']
        display_sim.data.qpos[2 * hand_qpos_dim: 2 * hand_qpos_dim + 7] = frame['obj_pose']
        display_sim.data.qpos[2 * hand_qpos_dim + 7:] = frame['e_obj_pose']

        display_sim_back.data.qpos[:hand_qpos_dim] = frame['hand_pose']
        display_sim_back.data.qpos[hand_qpos_dim: 2 * hand_qpos_dim] = frame['e_hand_pose']
        display_sim_back.data.qpos[2 * hand_qpos_dim: 2 * hand_qpos_dim + 7] = frame['obj_pose']
        display_sim_back.data.qpos[2 * hand_qpos_dim + 7:] = frame['e_obj_pose']

        display_sim.forward()
        display_sim_back.forward()

        viewer.render()
        viewer2.render()

        # if frame['reset']:
        #     viewer.add_marker(pos=np.array([0, 0, 0.8]),
        #                       label='reset',
        #                       rgba=np.array([1.0, 0, 0, 1.0]),
        #                       size=[0.01, 0.01, 0.01])
        #
        # # Add markers
        # if is_show_contact:
        #     contact_arr = frame['contact_points']
        #     ref_contact_arr = []
        #     for contact in display_sim.data.contact[:display_sim.data.ncon]:
        #         g1, g2 = contact.geom1, contact.geom2
        #         if ref_hand_geom_range[0] <= g1 <= ref_hand_geom_range[1] \
        #                 and ref_obj_geom_range[0] <= g1 <= ref_obj_geom_range[1]:
        #             ref_contact_arr.append(contact.pos.copy())
        #     viewer._markers[:] = []
        #     for c in contact_arr:
        #         viewer.add_marker(pos=c[:3],
        #                           label='',
        #                           rgba=np.array([1.0, 0, 0, 1.0]),
        #                           size=[0.005, 0.005, 0.005])
        #     for c in ref_contact_arr:
        #         viewer.add_marker(pos=c,
        #                           label='',
        #                           rgba=np.array([0.0, 0, 1.0, 1.0]),
        #                           size=[0.005, 0.005, 0.005])


class RLTest:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cuda:0'
        self.reset_threshold = 12

        # setup sim model
        self.sim_model_path = 'sim_model_temp.xml'
        sim_xml = self.load_sim_xml()
        with open(self.sim_model_path, 'w') as f:
            f.write(sim_xml)

        # setup data loader model
        self.data_load_model = mjpy.load_model_from_path(self.sim_model_path)
        self.data_load_sim = mjpy.MjSim(self.data_load_model)
        self.joint_lower_limit = self.data_load_model.jnt_range[:-1, 0]
        self.joint_upper_limit = self.data_load_model.jnt_range[:-1, 1]
        self.body_names = self.data_load_model.body_names
        self.hand_body_names = [name for name in self.body_names if name.startswith("link")]
        self.hand_body_idx = [self.data_load_model.body_name2id(name) for name in self.hand_body_names]

        # load policy net
        self.action_dim = 32
        self.state_dim = 617    # 617  # 593 # 217 # 225
        self.policy_net = PolicyGaussian(cfg, action_dim=self.action_dim, state_dim=self.state_dim)
        self.value_net = Value(MLP(self.state_dim, cfg.value_hsize, cfg.value_htype))
        cp_path = f"{cfg.model_dir}/iter_{cfg.epoch:04d}.p"
        print("loading model from checkpoint: %s" % cp_path)
        model_cp = CustomUnpickler(open(cp_path, "rb")).load()
        self.policy_net.load_state_dict(model_cp["policy_dict"])
        self.value_net.load_state_dict(model_cp["value_dict"])
        self.running_state = model_cp["running_state"]
        self.policy_net = self.policy_net.to(self.device)
        self.value_net = self.value_net.to(self.device)

        # setup frame
        self.frame_buf = []
        self.obs = None
        self.env = None
        self.last_frame = None
        self.motion_freq = cfg.data_specs['motion_freq']

        # init render
        worker = multiprocessing.Process(target=render_worker, args=(frame_queue, cfg))
        worker.start()

    def load_sim_xml(self):
        hand_model_file = self.cfg.mujoco_model_file
        obj_model_file = self.cfg.data_specs['obj_fn']

        mj_xml = MujocoXML(hand_model_file)
        mj_xml.merge(MujocoXML(obj_model_file))
        return mj_xml.get_xml()

    def make_frame(self, hand_pose, obj_pose):
        # clamp
        ndof = hand_pose.shape[0]
        hand_dof = np.clip(hand_pose, self.joint_lower_limit, self.joint_upper_limit)
        self.data_load_sim.data.qpos[:ndof] = hand_pose
        self.data_load_sim.forward()
        body_pos = self.data_load_sim.data.body_xpos[self.hand_body_idx]
        body_quat = self.data_load_sim.data.body_xquat[self.hand_body_idx]

        hand_dof_vel = np.zeros_like(hand_dof)
        obj_vel = np.zeros(3)
        obj_angle_vel = np.zeros(3)
        if self.last_frame is not None:
            # compute velocity
            hand_dof_vel = hand_dof - self.last_frame['hand_dof_seq']

            # align rotation to [-pi, pi]
            rot_qpos_err = hand_dof_vel[3:6].copy()
            while np.any(rot_qpos_err > np.pi):
                rot_qpos_err[rot_qpos_err > np.pi] -= 2 * np.pi
            while np.any(rot_qpos_err < -np.pi):
                rot_qpos_err[rot_qpos_err < -np.pi] += 2 * np.pi
            hand_dof_vel[3:6] = rot_qpos_err[:]

            hand_dof_vel = hand_dof_vel * self.motion_freq

            # compute object velocity
            obj_vel = obj_pose[0:3] - self.last_frame['obj_pose_seq'][0:3]
            obj_vel = obj_vel * self.motion_freq

            # compute object angle velocity
            obj_quat = obj_pose[3:]
            prev_obj_quat = self.last_frame['obj_pose_seq'][3:]
            rot_mat = quaternion_to_matrix(torch.Tensor(obj_quat))
            prev_rot_mat = quaternion_to_matrix(torch.Tensor(prev_obj_quat))
            relative_rot_seq = torch.matmul(rot_mat, torch.transpose(prev_rot_mat, 0, 1))
            obj_angle_vel = matrix_to_axis_angle(relative_rot_seq).numpy() * self.motion_freq

        frame = {
            "hand_dof_seq": hand_dof,
            "hand_dof_vel_seq": hand_dof_vel,
            "obj_vel_seq": obj_vel,
            "obj_angle_vel_seq": obj_angle_vel,
            "obj_pose_seq": obj_pose,
            "body_pos_seq": body_pos,
            "body_quat_seq": body_quat
        }

        self.last_frame = frame
        return frame

    def add_frame(self, frame):
        if self.env is None:
            self.frame_buf.append(frame)
            if len(self.frame_buf) == self.cfg.future_w_size + 1:
                self.env = HandObjMimicTest(self.cfg, self.frame_buf, self.sim_model_path, self.cfg.data_specs,
                                            mode="test")
                self.obs = self.env.reset()
                frame_queue.put({
                    'e_hand_pose': self.env.get_expert_hand_qpos(),
                    'hand_pose': self.env.get_hand_qpos(),
                    'e_obj_pose': self.env.get_expert_obj_pose(),
                    'obj_pose': self.env.get_obj_qpos(),
                    'contact_points': np.array([]),
                    'reset': False
                })
                # self.update_viewer()
        else:
            is_reset = self.step(frame)
            frame_queue.put({
                'e_hand_pose': self.env.get_expert_hand_qpos(),
                'hand_pose': self.env.get_hand_qpos(),
                'e_obj_pose': self.env.get_expert_obj_pose(),
                'obj_pose': self.env.get_obj_qpos(),
                'contact_points': np.concatenate([np.stack(x) for x in self.env.contact_frame_arr if len(x) != 0])[:, :3],
                # self.env.avg_cps.copy()
                'reset': is_reset
            })
            # self.update_viewer()

    def step(self, frame):
        with torch.no_grad():
            if self.running_state is not None:
                self.obs = self.running_state(self.obs, update=False)
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            value = self.value_net(obs_tensor).item()

            action = self.policy_net.select_action(obs_tensor, mean_action=True)[0].cpu().numpy()
            action = action.astype(np.float64)
            self.env.insert_new_frame(frame)
            self.obs, reward, done, info = self.env.step(action)

            if value < self.reset_threshold:
                self.obs = self.env.reset(True)

            return value < self.reset_threshold
            # obs = running_state(obs, update=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)
