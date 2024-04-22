# coding=utf-8
import json
import os

os.add_dll_directory(os.path.abspath("mujoco210//bin"))
import os.path
import time

import cv2
import trimesh
import numpy as np
import torch
import tqdm
import pickle
import transforms
import mujoco_py as mjpy

from uhc.data_loaders.mjxml.MujocoXML import MujocoXML
from manopth.manolayer import ManoLayer
from uhc.utils.transforms import *

keypoint_id = [
    25,  # root
    19, 18, 17, 16,  # thumb (base -> end)
    11, 10, 9, 8,  # index
    15, 14, 13, 12,  # middle
    3, 2, 1, 0,  # ring
    7, 6, 5, 4,  # little
]

kp_colors = [
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.6], [0.0, 0.0, 0.4],
    [1.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.6, 0.0], [0.0, 0.4, 0.0],
    [1.0, 1.0, 0.0], [0.8, 0.8, 0.0], [0.6, 0.6, 0.0], [0.4, 0.4, 0.0],
    [0.0, 1.0, 1.0], [0.0, 0.8, 0.8], [0.0, 0.6, 0.6], [0.0, 0.4, 0.4],
]

hand_dof_reduce_map = [0, 1, 2, 3, 4, 5,  # global
                       7, 8, 11, 14,  # index
                       25, 26, 29, 32,  # middle
                       16, 17, 20, 23,  # little
                       34, 35, 38, 41,  # ring
                       42, 43, 44, 45, 46, 47, 49]  # thumb

z_offset = 0.5 + 0.0839

calib_mat = np.array([[[0.8706465, -0.3992058, 0.28741852, -0.55512325],
                       [-0.48889736, -0.63768294, 0.59526466, -0.48630932],
                       [-0.05435123, -0.65878325, -0.75036708, 0.69956912],
                       [0., 0., 0., 1.]],

                      [[0.99665575, -0.07929539, -0.01973728, -0.29465306],
                       [-0.0466164, -0.75011281, 0.65966469, -0.63609843],
                       [-0.06711357, -0.65653853, -0.75130071, 0.70468118],
                       [0., 0., 0., 1.]],

                      [[0.08363583, -0.58672732, 0.80545406, -0.67241301],
                       [-0.9964331, -0.04012605, 0.07423712, 0.18207443],
                       [-0.01123732, -0.80878991, -0.58799053, 0.58616261],
                       [0., 0., 0., 1.]],

                      [[-0.99068907, 0.05912136, -0.12263691, 0.43946636],
                       [0.13372166, 0.59172123, -0.79497453, 0.73309331],
                       [0.02556689, -0.8039718, -0.59411756, 0.5785325],
                       [0., 0., 0., 1.]],

                      [[0.91584457, -0.24949845, 0.31460968, -0.50187081],
                       [-0.4011617, -0.53486124, 0.74363478, -0.78245444],
                       [-0.01726319, -0.80726323, -0.58993898, 0.58875561],
                       [0., 0., 0., 1.]],

                      [[0.98144634, -0.12443962, 0.14586937, -0.2642713],
                       [-0.19133033, -0.5860936, 0.7873291, -0.90845764],
                       [-0.01248184, -0.8006305, -0.59902849, 0.59714903],
                       [0., 0., 0., 1.]],

                      [[0.96898761, 0.15201944, -0.19481555, 0.1174773],
                       [0.24707051, -0.58204961, 0.77470924, -0.88751464],
                       [0.00437855, -0.79881683, -0.60155838, 0.59790447],
                       [0., 0., 0., 1.]],

                      [[0.91695315, -0.24014713, 0.3186319, -0.52161587],
                       [-0.39887657, -0.53228471, 0.7467066, -0.78873758],
                       [-0.00971658, -0.81178976, -0.58386921, 0.5784907],
                       [0., 0., 0., 1.]],

                      [[0.99903504, 0.04185076, -0.01332329, -0.18386524],
                       [0.03595802, -0.60519204, 0.79526694, -0.886033],
                       [0.02521938, -0.79497862, -0.60611292, 0.59729338],
                       [0., 0., 0., 1.]],

                      [[-0.87362345, 0.27338946, -0.40254217, 0.56758959],
                       [0.48611861, 0.52722589, -0.69693726, 0.58162745],
                       [0.02169534, -0.804544, -0.59349658, 0.58539987],
                       [0., 0., 0., 1.]]])

device = 'cuda:0'
mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=True, ncomps=45, flat_hand_mean=False)
mano_layer = mano_layer.to(device)
c_v, c_j = mano_layer(th_pose_coeffs=torch.zeros((1, 48)).to(device),
                      th_betas=torch.zeros((1, 10)).to(device),
                      th_trans=torch.zeros((1, 3)).to(device))


def load_motion(hand_trans, hand_rot, hand_theta, obj_trans, obj_rot):
    # transfer hand pca pose to dofs
    hand_root_pos = c_j[0][0] / 1000
    pred_theta = hand_theta.mm(mano_layer.th_selected_comps) + mano_layer.th_hands_mean
    pred_theta = torch.cat([torch.zeros(hand_theta.size(0), 3).to(device), pred_theta], dim=-1)
    global_rot = matrix_to_axis_angle(hand_rot)
    pred_theta[:, :3] = global_rot
    mano_trans = hand_trans

    hand_mid_pose = pred_theta[:, 12:21].clone()
    hand_little_pose = pred_theta[:, 21:30].clone()
    pred_theta[:, 12:21] = hand_little_pose
    pred_theta[:, 21:30] = hand_mid_pose

    # Axisangle to Euler
    pred_theta = pred_theta.view(pred_theta.shape[0], -1, 3)
    pred_theta = transforms.axis_angle_to_matrix(pred_theta)
    pred_theta = transforms.matrix_to_euler_angles(pred_theta, "XYZ")
    pred_theta = pred_theta.view(pred_theta.shape[0], pred_theta.shape[1] * pred_theta.shape[2])
    hand_pose_seq = torch.cat([mano_trans, pred_theta], dim=-1)

    # Obj pose
    obj_quat = matrix_to_quaternion(obj_rot)
    obj_pose_seq = torch.cat([obj_trans, obj_quat], dim=-1)

    return hand_pose_seq[:], obj_pose_seq[:]


def add_obj(hand_fn, obj_fn):
    if obj_fn is None:
        raise ValueError("obj_fn is None!")
    mj_xml = MujocoXML(hand_fn)
    mj_xml.merge(MujocoXML(obj_fn))
    mj_xml.get_xml()
    return mj_xml.get_xml()


def show_motion(all_seqs):
    hand_fn = 'assets/hand_model/manohand_reduce2/mano_hand_tip.xml'
    obj_fn = 'assets/Hotrack/mustard_bottle.xml'
    model_fn = 'dataset_model_tmp2.xml'
    with open(model_fn, 'w') as f:
        model_xml = add_obj(hand_fn, obj_fn)
        f.write(model_xml)
    model = mjpy.load_model_from_path(model_fn)
    # model = mjpy.load_model_from_xml(add_obj(hand_fn, obj_fn))
    sim = mjpy.MjSim(model)
    viewer = mjpy.MjViewer(sim)
    ndof = all_seqs[0]['hand_pose_seq'].shape[1]

    t = 0
    for i, seq in enumerate(all_seqs[:]):
        hand_pose_seq = seq['gt_hand_pose_seq']
        obj_pose_seq = seq['gt_obj_pose_seq']

        joint_upper_limit = model.jnt_range[:-1, 1]
        joint_lower_limit = model.jnt_range[:-1, 0]
        hand_pose_seq = np.clip(hand_pose_seq, joint_lower_limit, joint_upper_limit)

        t = 0
        print('Seq ' + str(i))
        while True:
            print(str(t) + '/' + str(hand_pose_seq.shape[0]))
            sim.data.qpos[:ndof] = hand_pose_seq[t, :]
            sim.data.qpos[ndof:] = obj_pose_seq[t, :]
            sim.forward()
            viewer.add_marker(pos=np.array([0, 0, 0.8]),
                              label=str(i),
                              rgba=np.array([1.0, 0, 0, 1.0]),
                              size=[0.01, 0.01, 0.01])
            viewer.render()
            t += 1
            if t >= hand_pose_seq.shape[0]:
                break


def calib_global_pose(hand_trans, hand_rot, obj_trans, obj_rot, cam_ext, cam_to_world):
    obj_trans = torch.matmul(obj_trans, cam_ext[:3, :3].t()) + cam_ext[:3, 3]
    obj_rot = torch.matmul(cam_ext[:3, :3], obj_rot)
    hand_trans = torch.matmul(hand_trans, cam_ext[:3, :3].to(device).t()) + cam_ext[:3, 3].to(device)
    hand_rot = torch.matmul(cam_ext[:3, :3].to(device), hand_rot)

    # calib camera coordinate to world coordinate
    obj_trans = torch.matmul(obj_trans, cam_to_world[:3, :3].t()) + cam_to_world[:3, 3]
    obj_rot = torch.matmul(cam_to_world[:3, :3], obj_rot)
    hand_trans = torch.matmul(hand_trans, cam_to_world[:3, :3].to(device).t()) + cam_to_world[:3, 3].to(device)
    hand_rot = torch.matmul(cam_to_world[:3, :3].to(device), hand_rot)

    # Add z offset
    obj_trans[:, 2] += z_offset
    hand_trans[:, 2] += z_offset

    return hand_trans, hand_rot, obj_trans, obj_rot


if __name__ == '__main__':
    data_root = 'sample_data/Hotrack/bottle'
    data_fn = 'sample_data/Hotrack/bottle/bottle_seq_selected.pkl'
    seq_name = 'dexycb_bottle_seq'

    is_show_motion = False
    is_dump_motion = True

    all_seqs = []
    problem_seq_idx = {39, 51, 58, 94, 115, 132, 137}
    test_split_sub_idx = 8
    test_split_idx = -1
    raw_seq = pickle.load(open(data_fn, 'rb'))
    for i, seq in enumerate(raw_seq):
        if i in problem_seq_idx:
            continue
        print(str(i) + ': ' + seq['name'])
        obj_trans = torch.stack([p['translation'].squeeze() for p in seq['pred_obj_pose']])
        obj_rot = torch.stack([p['rotation'].squeeze() for p in seq['pred_obj_pose']])
        hand_trans = torch.stack([p['translation'].squeeze() for p in seq['pred_hand_pose']['global_pose']]).to(
            device).to(torch.float32)
        hand_rot = torch.stack([p['rotation'].squeeze() for p in seq['pred_hand_pose']['global_pose']]).to(
            device).to(torch.float32)
        hand_theta = torch.stack([p.squeeze() for p in seq['pred_hand_pose']['MANO_theta']]).to(device).to(
            torch.float32)

        gt_obj_trans = torch.stack([p['translation'].squeeze() for p in seq['gt_obj_pose']])
        gt_obj_rot = torch.stack([p['rotation'].squeeze() for p in seq['gt_obj_pose']])
        gt_hand_trans = torch.stack([p['translation'].squeeze() for p in seq['gt_hand_pose']]).to(device).to(
            torch.float32)
        gt_hand_mano_trans = torch.stack([p['mano_trans'].squeeze() for p in seq['gt_hand_pose']]).to(device).to(
            torch.float32)
        gt_hand_rot = torch.stack([p['rotation'].squeeze() for p in seq['gt_hand_pose']]).to(device).to(torch.float32)
        gt_hand_theta = torch.stack([p['mano_pose'].squeeze() for p in seq['gt_hand_pose']]).to(device).to(
            torch.float32)
        gt_hand_theta = gt_hand_theta[:, 3:]

        # Camera coordinate to calib camera coordinate
        cam_ext = seq['extern']
        sub_str = seq['name'].split('+')[0]
        sub_idx = sub_str.split('-')[-1]
        sub_idx = int(sub_idx) - 1
        if sub_idx >= test_split_sub_idx and test_split_idx < 0:
            test_split_idx = len(all_seqs)
        cam_to_world = torch.Tensor(calib_mat[sub_idx])
        hand_trans, hand_rot, obj_trans, obj_rot = calib_global_pose(hand_trans, hand_rot, obj_trans, obj_rot, cam_ext, cam_to_world)
        gt_hand_trans, gt_hand_rot, gt_obj_trans, gt_obj_rot = calib_global_pose(gt_hand_trans, gt_hand_rot, gt_obj_trans, gt_obj_rot, cam_ext, cam_to_world)

        hand_pose_seq, obj_pose_seq = load_motion(hand_trans, hand_rot, hand_theta, obj_trans, obj_rot)
        gt_hand_pose_seq, gt_obj_pose_seq = load_motion(gt_hand_trans, gt_hand_rot, gt_hand_theta, gt_obj_trans, gt_obj_rot)
        hand_pose_seq = hand_pose_seq[:, hand_dof_reduce_map]
        gt_hand_pose_seq = gt_hand_pose_seq[:, hand_dof_reduce_map]
        all_seqs.append({
            "hand_pose_seq": hand_pose_seq.cpu().numpy(),
            "obj_pose_seq": obj_pose_seq.cpu().numpy(),
            "gt_hand_pose_seq": gt_hand_pose_seq.cpu().numpy(),
            "gt_obj_pose_seq": gt_obj_pose_seq.cpu().numpy()
        })

    if is_show_motion:
        show_motion(all_seqs)

    if is_dump_motion:
        # split train and test set
        train_set = all_seqs[:test_split_idx + 1]   # add one for validation
        test_set = all_seqs[test_split_idx:]
        pickle.dump({seq_name: train_set}, open(os.path.join(data_root, seq_name + '_train' + '.pkl'), 'wb'))
        pickle.dump({seq_name: test_set}, open(os.path.join(data_root, seq_name + '_test' + '.pkl'), 'wb'))
