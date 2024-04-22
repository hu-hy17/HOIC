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

import mujoco_py as mjpy

from uhc.data_loaders.dataset_grab_single import DatasetGRABSingle
from uhc.data_loaders.dataset_grab import DatasetGRAB
from uhc.envs.hand_im_display import HandMimicDisplayEnv
from uhc.envs.hand_im import HandMimicEnv, hand_mimic_reward
from uhc.envs.ho_im import HandObjMimic, ho_mimic_reward
from uhc.khrylib.rl.core import PolicyGaussian
from uhc.utils.tools import CustomUnpickler

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

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
    hand_model.merge(obj_model)
    ref_obj_model.worldbody.getchildren()[0].attrib['name'] = 'ref_' + ref_obj_model.worldbody.getchildren()[0].attrib['name']
    ref_obj_model.worldbody.getchildren()[0].getchildren()[0].attrib['material'] = "object_ref"
    ref_obj_model.worldbody.getchildren()[0].getchildren()[1].attrib['material'] = "object_ref"

    hand_model.merge(ref_obj_model)
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
    data_loader = DatasetGRAB(cfg.mujoco_model_file, cfg.data_specs)
    expert_seq = data_loader.load_seq(2)
    if cfg.data_specs["with_obj"]:
        env = HandObjMimic(cfg, expert_seq, data_loader.model_xml, cfg.data_specs, mode="test")
    else:
        env = HandMimicEnv(cfg, expert_seq, cfg.data_specs, mode="test")

    obs = env.reset()

    # load policy net
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    policy_net = PolicyGaussian(cfg, action_dim=action_dim, state_dim=state_dim)
    cp_path = f"{cfg.model_dir}/iter_{cfg.epoch:04d}.p"
    print("loading model from checkpoint: %s" % cp_path)
    model_cp = CustomUnpickler(open(cp_path, "rb")).load()
    policy_net.load_state_dict(model_cp["policy_dict"])
    running_state = model_cp["running_state"]

    pred_qpos_seq = []
    pred_qpos_seq.append(env.data.qpos.copy())
    gt_qpos_seq = expert_seq["hand_dof_seq"].copy()
    gt_obj_pose_seq = expert_seq["obj_pose_seq"].copy()
    gt_qpos_seq[:, 0:2] += 0.2
    gt_obj_pose_seq[:, 0:2] += 0.2

    with torch.no_grad():
        while True:
            if running_state is not None:
                obs = running_state(obs, update=False)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy_net.select_action(obs_tensor, mean_action=True)[0].cpu().numpy()
            action = action.astype(np.float64)
            obs, reward, done, info = env.step(action)

            pred_qpos_seq.append(env.data.qpos.copy())

            if cfg.data_specs["with_obj"]:
                print(ho_mimic_reward(env, obs, action, info))
            else:
                print(hand_mimic_reward(env, obs, action, info))
            if done:
                break

        # display seq
        viewer = mjpy.MjViewer(display_sim)
        for idx in range(0, 10000):
            cur_t = idx % len(pred_qpos_seq)
            display_sim.data.qpos[:env.hand_qpos_dim] = pred_qpos_seq[cur_t][:env.hand_qpos_dim]
            display_sim.data.qpos[env.hand_qpos_dim: 2 * env.hand_qpos_dim] = gt_qpos_seq[cur_t]
            if cfg.data_specs["with_obj"]:
                display_sim.data.qpos[2 * env.hand_qpos_dim: 2 * env.hand_qpos_dim + 7] = pred_qpos_seq[cur_t][env.hand_qpos_dim:]
                display_sim.data.qpos[2 * env.hand_qpos_dim + 7:] = gt_obj_pose_seq[cur_t]
            display_sim.forward()
            viewer.render()
            time.sleep(0.03)
