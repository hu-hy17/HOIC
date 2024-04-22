import glob
import time
import torch
import numpy as np
import os
import pickle
import json

import tqdm

import transforms
import mujoco_py as mjpy
from manopth.manolayer import ManoLayer

from uhc.data_loaders.mjxml.MujocoXML import MujocoXML

torch.set_printoptions(threshold=np.inf)

hand_dof_reduce_map = [0, 1, 2, 3, 4, 5,    # global
                       7, 8, 11, 14,        # index
                       25, 26, 29, 32,      # middle
                       16, 17, 20, 23,      # little
                       34, 35, 38, 41,      # ring
                       42, 43, 44, 45, 46, 47, 49]  # thumb

# clip_idx = [
#     [[0, 600], [760, 960], [1200, 1900]],
#     [[0, 2400]],
#     [[0, 1100], [1600, 1800]],
#     [[0, 1600]],
#     [[0, 700], [1250, 1750]]
# ]

clip_idx = [
    [[120, 600], [760, 960], [1200 + 140, 1900]],
    [[0+180, 2400 - 200]],
    [[0+100, 1100], [1600, 1800]],
    [[0+180, 1600]],
    [[0+180, 700], [1250, 1750]]
]

def seq_to_tensor(hand_rot, hand_trans, hand_pose, obj_rot, obj_trans):
    hand_rot_tensor = torch.tensor(hand_rot, dtype=torch.float)
    hand_trans_tensor = torch.tensor(hand_trans, dtype=torch.float)
    hand_pose_tensor = torch.tensor(hand_pose, dtype=torch.float)
    obj_rot_tensor = torch.tensor(obj_rot, dtype=torch.float)
    obj_trans_tensor = torch.tensor(obj_trans, dtype=torch.float)

    return torch.cat([hand_rot_tensor, hand_pose_tensor, hand_trans_tensor], dim=1), \
        torch.cat([obj_trans_tensor, obj_rot_tensor], dim=1)


def dump_motion(hand_motion: torch.Tensor, obj_motion: torch.Tensor, hand_kp, name: str):
    trajectory_data = {'hand_dof': hand_motion.cpu().detach().numpy().tolist(),
                       'obj_pose': obj_motion.cpu().detach().numpy().tolist(),
                       'hand_kp': hand_kp.cpu().detach().numpy().tolist()}
    traj_file = open('trajectory/' + name + '.json', 'w')
    json.dump(trajectory_data, traj_file, indent=4)
    traj_file.close()


def load_motion(seq_path):
    seq_info = np.load(seq_path, allow_pickle=True)
    seq_info = {k: seq_info[k].item() for k in seq_info.files}
    hand_info = seq_info['rhand']['params']
    obj_info = seq_info['object']['params']

    hand_pose_seq, obj_pose_seq = seq_to_tensor(hand_info['global_orient'],
                                                hand_info['transl'],
                                                hand_info['fullpose'],
                                                obj_info['global_orient'],
                                                obj_info['transl'])

    # transfer hand pca pose to dofs
    mano_layer = ManoLayer(mano_root='data/mano', use_pca=False, ncomps=45, flat_hand_mean=True)
    hand_v, hand_j = mano_layer(th_pose_coeffs=hand_pose_seq[:, 0:48],
                                th_trans=hand_pose_seq[:, 48:51],
                                th_betas=torch.zeros((hand_pose_seq.shape[0], 10)))
    th_v_shaped = torch.matmul(mano_layer.th_shapedirs,
                               mano_layer.th_betas.transpose(1, 0)).permute(2, 0, 1) + mano_layer.th_v_template
    th_j = torch.matmul(mano_layer.th_J_regressor, th_v_shaped)
    hand_root_pos = th_j[0][0]
    hand_pose_seq[:, 48:51] += hand_root_pos
    hand_mid_pose = hand_pose_seq[:, 12:21].clone()
    hand_little_pose = hand_pose_seq[:, 21:30].clone()
    hand_pose_seq[:, 12:21] = hand_little_pose
    hand_pose_seq[:, 21:30] = hand_mid_pose
    # hand_verts, hand_joints = mano_layer(hand_pose_seq[:, 0:48], torch.zeros(num_frames, 10), hand_pose_seq[:, 48:])

    # align coordinate
    z_offset = 0.5 + 0.0683  # 0.0683-bottle 0.025-airplane
    obj_rot_mat = transforms.axis_angle_to_matrix(obj_pose_seq[:, 3:6])
    obj_init_rot_mat = obj_rot_mat[0].clone()
    obj_init_pos = obj_pose_seq[0, 0:3].clone()
    obj_init_rot_inv = obj_init_rot_mat.transpose(0, 1)

    # align object
    obj_rot_mat = torch.matmul(obj_init_rot_inv, obj_rot_mat)
    obj_pose_seq[:, 3:6] = transforms.matrix_to_axis_angle(obj_rot_mat)
    obj_pose_seq[:, 0:3] = (obj_pose_seq[:, 0:3] - obj_init_pos).mm(obj_init_rot_mat)

    # align hand
    hand_pose_seq[:, 48:] = (hand_pose_seq[:, 48:] - obj_init_pos).mm(obj_init_rot_mat)
    hand_global_rot = hand_pose_seq[:, :3].clone()
    hand_global_rot = torch.matmul(obj_init_rot_inv, transforms.axis_angle_to_matrix(hand_global_rot))
    hand_global_rot = transforms.matrix_to_axis_angle(hand_global_rot)
    hand_pose_seq[:, :3] = hand_global_rot

    # z-axis offset
    obj_pose_seq[:, 2] += z_offset
    hand_pose_seq[:, 50] += z_offset

    # new_dist = torch.norm(obj_pose_seq[:, 0:3] - hand_pose_seq[:, 48:51], dim=-1)

    # print(torch.max(torch.abs(new_dist - dist)))

    # # Get axis angle from PCA components and coefficients
    # th_hand_pose_coeffs = hand_pose_seq[:, 3:48].clone()
    # th_full_hand_pose = th_hand_pose_coeffs.mm(mano_layer.th_selected_comps)
    # th_full_pose = torch.cat([hand_pose_seq[:, :3], mano_layer.th_hands_mean + th_full_hand_pose], 1)
    # th_full_pose = th_full_pose.view(num_frames, -1, 3)

    # Transform axis angle to euler angle
    th_full_pose = hand_pose_seq[:, 0:48].view(hand_pose_seq.shape[0], -1, 3)
    th_full_pose = transforms.axis_angle_to_matrix(th_full_pose)
    th_full_pose = transforms.matrix_to_euler_angles(th_full_pose, "XYZ")
    th_full_pose = th_full_pose.view(hand_pose_seq.shape[0], th_full_pose.shape[1] * th_full_pose.shape[2])
    # hand_pose_seq = torch.cat([hand_pose_seq[:, 48:], hand_pose_seq[:, 0:48]], dim=1)
    hand_pose_seq = torch.cat([hand_pose_seq[:, 48:], th_full_pose], dim=1)

    # Transform object pose
    obj_rot_quat = transforms.axis_angle_to_quaternion(obj_pose_seq[:, 3:6])
    obj_trans = obj_pose_seq[:, 0:3]
    obj_pose_seq = torch.cat([obj_trans, obj_rot_quat], 1)

    return hand_pose_seq[:], obj_pose_seq[:], hand_j / 1000


def add_obj(hand_fn, obj_fn):
    if obj_fn is None:
        raise ValueError("obj_fn is None!")
    mj_xml = MujocoXML(hand_fn)
    mj_xml.merge(MujocoXML(obj_fn))
    mj_xml.get_xml()
    return mj_xml.get_xml()


if __name__ == '__main__':
    show_motion = True
    dump_motion = True

    if show_motion:
        hand_fn = 'assets/hand_model/manohand_reduce2/mano_hand.xml'
        obj_fn = 'assets/grab/waterbottle.xml'
        model = mjpy.load_model_from_xml(add_obj(hand_fn, obj_fn))
        sim = mjpy.MjSim(model)
        viewer = mjpy.MjViewer(sim)

    motion_seqs = []

    root_path = "data/grab_dataset/data/grab"
    seq_name = "waterbottle_lift"  # waterbottle_lift airplane_fly_1
    seq_list = glob.glob(root_path + '/*/' + seq_name + '.npz')

    for idx in tqdm.tqdm(range(len(seq_list))):
        full_hand_pose_seq, full_obj_pose_seq, _ = load_motion(seq_list[idx])
        full_hand_pose_seq = full_hand_pose_seq[:, hand_dof_reduce_map]
        ndof = full_hand_pose_seq.shape[1]
        for ci in range(len(clip_idx[idx])):
            clip_s = clip_idx[idx][ci][0]
            clip_e = clip_idx[idx][ci][1]
            hand_pose_seq = full_hand_pose_seq[clip_s: clip_e]
            obj_pose_seq = full_obj_pose_seq[clip_s: clip_e]
            motion_seqs.append({
                "hand_pose_seq": hand_pose_seq.numpy(),
                "obj_pose_seq": obj_pose_seq.numpy()
            })
            t = 0
            if show_motion:
                while True:
                    print(t)
                    sim.data.qpos[:ndof] = hand_pose_seq[t, :]
                    sim.data.qpos[ndof:] = obj_pose_seq[t, :]
                    sim.forward()
                    viewer.render()
                    t += 4
                    if t >= hand_pose_seq.shape[0]:
                        break
                    time.sleep(0.01)

    # dump motion
    if dump_motion:
        pickle.dump({seq_name: motion_seqs}, open('sample_data/grab/' + seq_name + '_short' + '.pkl', 'wb'))
