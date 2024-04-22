"""
File: /eval_copycat.py
Created Date: Monday June 7th 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Monday June 7th 2021 3:57:49 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
"""

import argparse

import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp
import os
sys.path.append(os.getcwd())
import mujoco_py as mjpy
import tqdm
import matplotlib.pyplot as plt

from uhc.data_loaders.dataset_grab_single import DatasetGRABSingle
from uhc.data_loaders.dataset_grab import DatasetGRAB
from uhc.envs.hand_im_display import HandMimicDisplayEnv
from uhc.envs.hand_im import HandMimicEnv
from uhc.envs.ho_reward import *
from uhc.envs.ho_im import HandObjMimic
from uhc.khrylib.rl.core import PolicyGaussian
from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.core.critic import Value
from uhc.utils.tools import CustomUnpickler
from uhc.utils.torch_utils import *

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np

from uhc.utils.flags import flags
from uhc.utils.config_utils.handmimic_config import Config
from uhc.data_loaders.mjxml.MujocoXML import MujocoXML
from uhc.utils.image_utils import write_frames_to_video
import wandb


def load_display_xml(cfg):
    hand_model_file = cfg.vis_model_file
    obj_model_file = cfg.data_specs['obj_fn']
    hand_model = MujocoXML(hand_model_file)
    obj_model = MujocoXML(obj_model_file)
    ref_obj_model = MujocoXML(obj_model_file)
    # ref_obj_model2 = MujocoXML(obj_model_file)

    ref_obj_model.worldbody.getchildren()[0].attrib['name'] += '_ref'
    ref_obj_model.worldbody.getchildren()[0].getchildren()[0].attrib['material'] = "object_ref"
    ref_obj_model.worldbody.getchildren()[0].getchildren()[1].attrib['material'] = "object_ref"
    ref_obj_model.worldbody.getchildren()[0].getchildren()[0].attrib['name'] += "_ref"
    ref_obj_model.worldbody.getchildren()[0].getchildren()[1].attrib['name'] += "_ref"
    # ref_obj_model2.worldbody.getchildren()[0].attrib['name'] = 'ref2_' + ref_obj_model.worldbody.getchildren()[0].attrib['name']
    # ref_obj_model2.worldbody.getchildren()[0].getchildren()[0].attrib['material'] = "object_ref2"
    # ref_obj_model2.worldbody.getchildren()[0].getchildren()[1].attrib['material'] = "object_ref2"

    hand_model.merge(obj_model)
    hand_model.merge(ref_obj_model)
    # hand_model.merge(ref_obj_model2)
    return hand_model.get_xml()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    # setup display
    if cfg.data_specs['with_obj']:
        display_xml = load_display_xml(cfg)
        display_model = mjpy.load_model_from_xml(display_xml)
        display_sim = mjpy.MjSim(display_model)
    else:
        display_model = mjpy.load_model_from_path(cfg.vis_model_file)
        display_sim = mjpy.MjSim(display_model)

    # setup env
    # data_loader = DatasetGRABSingle(cfg.mujoco_model_file, cfg.data_specs)
    data_loader = DatasetGRAB(cfg.mujoco_model_file, cfg.data_specs, noise=0)
    expert_seq = data_loader.load_seq(0, full_seq=True)
    if cfg.data_specs["with_obj"]:
        env = HandObjMimic(cfg, expert_seq, data_loader.model_xml, cfg.data_specs, mode="test")
    else:
        env = HandMimicEnv(cfg, expert_seq, cfg.data_specs, mode="test")

    # load policy net
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    policy_net = PolicyGaussian(cfg, action_dim=action_dim, state_dim=state_dim)
    value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
    cp_path = f"{cfg.model_dir}/iter_{cfg.epoch:04d}.p"
    print("loading model from checkpoint: %s" % cp_path)
    model_cp = CustomUnpickler(open(cp_path, "rb")).load()
    policy_net.load_state_dict(model_cp["policy_dict"])
    value_net.load_state_dict(model_cp["value_dict"])
    running_state = model_cp["running_state"]

    all_pred_qpos_seq = []
    all_ref_qpos_seq = []
    all_obj_ref_pose_seq = []
    all_gt_qpos_seq = []

    reward_func = reward_list[cfg.reward_type - 1]

    with torch.no_grad():
        for seq_idx in tqdm.tqdm(range(data_loader.seq_num)):
            # seq_idx = 0
            expert_seq = data_loader.load_seq(seq_idx, start_idx=0, full_seq=True)
            env.set_expert(expert_seq)
            pred_qpos_seq = []
            pred_joint_pos = []
            tot_reward_seq = []
            all_reward_seq = []
            value_seq = []
            pred_qpos_seq.append(env.data.qpos.copy())
            ref_qpos_seq = expert_seq["hand_dof_seq"].copy()
            ref_obj_pose_seq = expert_seq["obj_pose_seq"].copy()
            gt_qpos_seq = expert_seq["raw_hand_dof_seq"].copy()
            ref_joint_pos = expert_seq["body_pos_seq"].copy()
            gt_joint_pos = expert_seq["raw_body_pos_seq"].copy()
            obs = env.reset()
            pred_joint_pos.append(env.get_wbody_pos().copy().reshape(-1, 3))

            while True:
                if running_state is not None:
                    obs = running_state(obs, update=False)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = policy_net.select_action(obs_tensor, mean_action=True)[0].cpu().numpy()
                value = value_net(obs_tensor)
                action = action.astype(np.float64)
                obs, reward, done, info = env.step(action)
                tot_rew, all_rew = reward_func(env, obs, action, info)
                tot_reward_seq.append(tot_rew)
                all_reward_seq.append(all_rew)
                value_seq.append(value)

                pred_qpos_seq.append(env.data.qpos.copy())
                pred_joint_pos.append(env.get_wbody_pos().copy().reshape(-1, 3))

                if done:
                    break

            # Compute average reward
            tot_reward_seq = np.stack(tot_reward_seq)
            all_reward_seq = np.stack(all_reward_seq)

            print("Avg tot reward: %.4f" % np.mean(tot_reward_seq[:-1]))
            print("Avg all reward: ", np.mean(all_reward_seq, axis=0))

            # Compute MPJPE, object pos error and orientation error
            pred_qpos_seq = np.stack(pred_qpos_seq)
            pred_joint_pos = np.stack(pred_joint_pos)

            sim_step = pred_joint_pos.shape[0]
            gt_joint_pos = gt_joint_pos[:sim_step]
            ref_joint_pos = ref_joint_pos[:sim_step]
            ref_obj_pose_seq = ref_obj_pose_seq[:sim_step]

            mpjpe = np.mean(np.linalg.norm(pred_joint_pos - gt_joint_pos, axis=2), axis=0)[0]
            print("MPJPE(Pred Vs GT): %.4f" % mpjpe)
            mpjpe = np.mean(np.linalg.norm(ref_joint_pos - gt_joint_pos, axis=2))
            print("MPJPE(Ref Vs GT): %.4f" % mpjpe)
            obj_pos_err = pred_qpos_seq[:, env.ndof:env.ndof+3] - ref_obj_pose_seq[:, :3]
            obj_pos_err = np.linalg.norm(obj_pos_err, axis=1)
            print("Obj pos err: %.4f" % np.mean(obj_pos_err))
            pred_obj_rot = torch.Tensor(pred_qpos_seq[:, env.ndof+3:])
            gt_obj_rot = torch.Tensor(ref_obj_pose_seq[:, 3:])
            obj_quat_diff = quaternion_multiply_batch(gt_obj_rot, quaternion_inverse_batch(pred_obj_rot))
            obj_rot_diff = 2.0 * torch.arcsin(torch.clip(torch.norm(obj_quat_diff[:, 1:], dim=-1), 0, 1))
            obj_rot_diff = obj_rot_diff.cpu().numpy()
            print("Obj rot err: %.4f" % np.mean(obj_rot_diff))

            ref_qpos_seq[:, 0] += 0.2
            ref_obj_pose_seq[:, 0] += 0.2
            all_pred_qpos_seq.append(pred_qpos_seq)
            all_ref_qpos_seq.append(ref_qpos_seq)
            all_obj_ref_pose_seq.append(ref_obj_pose_seq)
            all_gt_qpos_seq.append(gt_qpos_seq)

            # plot value
            # plt.plot(value_seq)
            # plt.show()

        # display seq
        viewer = mjpy.MjViewer(display_sim)
        viewer2 = mjpy.MjViewer(display_sim)
        for seq_idx in range(10000):
            seq_idx = seq_idx % len(all_pred_qpos_seq)
            # seq_idx = 0
            # print(seq_idx)
            pred_qpos_seq = all_pred_qpos_seq[seq_idx]
            ref_qpos_seq = all_ref_qpos_seq[seq_idx]
            gt_obj_pose_seq = all_obj_ref_pose_seq[seq_idx]
            for idx in range(len(pred_qpos_seq)):
                cur_t = idx
                display_sim.data.qpos[:env.hand_qpos_dim] = pred_qpos_seq[cur_t][:env.hand_qpos_dim]
                display_sim.data.qpos[env.hand_qpos_dim: 2 * env.hand_qpos_dim] = ref_qpos_seq[cur_t]
                if cfg.data_specs["with_obj"]:
                    display_sim.data.qpos[2 * env.hand_qpos_dim: 2 * env.hand_qpos_dim + 7] = pred_qpos_seq[cur_t][env.hand_qpos_dim:]
                    display_sim.data.qpos[2 * env.hand_qpos_dim + 7:] = gt_obj_pose_seq[cur_t]
                display_sim.forward()
                viewer.render()
                viewer2.render()
                #  time.sleep(0.03)
