# coding=utf-8
import json
import os

import yaml

from uhc.utils.objpose_calib import obj_pose_calib

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

kp_colors = [
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.6], [0.0, 0.0, 0.4],
    [1.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.6, 0.0], [0.0, 0.4, 0.0],
    [1.0, 1.0, 0.0], [0.8, 0.8, 0.0], [0.6, 0.6, 0.0], [0.4, 0.4, 0.0],
    [0.0, 1.0, 1.0], [0.0, 0.8, 0.8], [0.0, 0.6, 0.6], [0.0, 0.4, 0.4],
]

hand_dof_map = [0, 1, 2, 3, 4, 5,  # global
                14, 13, 15, 16,  # index
                18, 17, 19, 20,  # middle
                26, 25, 27, 28,  # little
                22, 21, 23, 24,  # ring
                10, 9, 11, 12]  # thumb

reverse_dof_idx = [
    7, 11, 15, 19, 23
]

ref_obj_fn = None
debug = False
data_root = 'sample_data/SingleDepth/Banana/'
is_show_motion = False
is_dump_motion = True
test_only = True


def LoadData(fn, o2c):
    json_root = json.load(open(fn, 'r'))

    motion_seq_json = json_root['motion_info']
    start_idx = motion_seq_json[0]['frame_id']
    seq_len = motion_seq_json[-1]['frame_id'] - start_idx + 1

    obj_pose_seq = []
    thetas_seq = []
    conf_seq = []
    for i in range(seq_len):
        thetas = torch.Tensor(motion_seq_json[i]['thetas'])
        thetas_seq.append(thetas)
        obj_pose_seq.append(torch.Tensor(motion_seq_json[i]["obj_motion"]).view(4, 4))
        conf_seq.append(motion_seq_json[i]['conf'])

    thetas_seq = torch.stack(thetas_seq, dim=0)
    obj_pose_seq = torch.stack(obj_pose_seq, dim=0)
    obj_pose_seq[:, 0:3, 3] /= 1000
    obj_pose_seq[:, 1:3, :] *= -1
    obj_pose_seq = torch.matmul(obj_pose_seq, torch.tensor(o2c).to(torch.float32))
    obj_pos_seq = obj_pose_seq[:, 0:3, 3]
    obj_rot_seq = transforms.matrix_to_quaternion(obj_pose_seq[:, 0:3, 0:3])

    return thetas_seq, obj_pos_seq, obj_rot_seq, conf_seq


def LoadObj(vf, nf):
    verts = []
    norms = []
    # load canonical object
    f = open(vf)
    lines = f.readlines()
    for line in lines:
        coord = line.split(' ')
        verts.append(coord)
    f.close()
    verts = np.array(verts).astype(float) / 1000

    f = open(nf)
    lines = f.readlines()
    for line in lines:
        norm = line.split(' ')
        norms.append(norm)
    f.close()
    norms = np.array(norms).astype(float)
    faces = np.arange(0, verts.shape[0]).reshape(-1, 3)

    obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, process=False)
    obj2cano = obj_pose_calib(ref_obj_fn, vf, debug)
    obj_mesh.apply_transform(obj2cano)
    obj2cano = np.linalg.inv(obj2cano)

    # obj_mesh.export('assets/tracking/Visual/box.stl')
    # obj_mesh.simplify_quadric_decimation(64)
    # obj_mesh.export('assets/tracking/Collision/box.stl')

    return obj2cano


def load_motion(hand_pose_seq, obj_pose_seq):
    hand_pose_seq = hand_pose_seq[:, hand_dof_map]
    hand_pose_seq[:, 0:3] = hand_pose_seq[:, 0:3] / 1000

    obj_rot_mat = transforms.quaternion_to_matrix(obj_pose_seq[:, 3:7])
    obj_pos_seq = obj_pose_seq[:, 0:3].clone()

    # left hand -> right hand
    hand_rot_mat = transforms.euler_angles_to_matrix(hand_pose_seq[:, 3:6], "XYZ")
    hand_rot_mat[:, :, 0] *= -1
    hand_rot_mat[:, 2, :] *= -1
    hand_pose_seq[:, reverse_dof_idx] *= -1

    # Swap Y and Z axis
    obj_rot_mat = obj_rot_mat[:, [0, 2, 1], :]
    obj_rot_mat[:, 1, :] *= -1
    obj_pos_seq = obj_pos_seq[:, [0, 2, 1]]
    obj_pos_seq[:, 1] *= -1
    hand_rot_mat = hand_rot_mat[:, [0, 2, 1], :]
    hand_rot_mat[:, 1, :] *= -1
    hand_pose_seq[:, 0:3] = hand_pose_seq[:, [0, 2, 1]]

    obj_pose_seq[:, 0:3] = obj_pos_seq
    obj_pose_seq[:, 3:7] = transforms.matrix_to_quaternion(obj_rot_mat)
    hand_pose_seq[:, 3:6] = transforms.matrix_to_euler_angles(hand_rot_mat, "XYZ")

    # Translate hand and object
    obj_init_pos = obj_pose_seq[0, 0:3].clone()
    obj_pose_seq[:, 0:3] = obj_pose_seq[:, 0:3] - obj_init_pos
    hand_pose_seq[:, 0:3] = hand_pose_seq[:, 0:3] - obj_init_pos

    # z-axis offset
    z_offset = 0.7
    obj_pose_seq[:, 2] += z_offset
    hand_pose_seq[:, 2] += z_offset

    return hand_pose_seq[:], obj_pose_seq[:], obj_init_pos


def add_obj(hand_fn, obj_fn):
    if obj_fn is None:
        raise ValueError("obj_fn is None!")
    mj_xml = MujocoXML(hand_fn)
    mj_xml.merge(MujocoXML(obj_fn))
    mj_xml.get_xml()
    return mj_xml.get_xml()


def symmetric_aug(obj_name, ori_obj_seq, ori_hand_seq, ori_confs, ori_init_obj_pos, all_seqs):
    if obj_name == 'box':
        ori_pos = ori_obj_seq[:, :3]
        ori_rot = ori_obj_seq[:, 3:]
        ori_rot = transforms.quaternion_to_matrix(torch.Tensor(ori_rot))
        for a in range(0, 3):
            aug_obj_rot = ori_rot * -1
            aug_obj_rot[:, :, a] *= -1
            aug_obj_rot = transforms.matrix_to_quaternion(aug_obj_rot)
            all_seqs.append({
                "hand_pose_seq": ori_hand_seq,
                "obj_pose_seq": np.concatenate([ori_pos, aug_obj_rot.numpy()], axis=1),
                "conf": ori_confs,
                "obj_init_pos": ori_init_obj_pos,
            })
    elif obj_name == 'bottle' or obj_name == 'cup':
        ori_pos = ori_obj_seq[:, :3]
        ori_rot = ori_obj_seq[:, 3:]
        ori_rot = transforms.quaternion_to_matrix(torch.Tensor(ori_rot))
        for a in range(0, 3):
            rot_vec = torch.Tensor([0, 0, 90 * (a + 1) / torch.pi])
            aug_obj_rot = torch.matmul(ori_rot, transforms.axis_angle_to_matrix(rot_vec))
            aug_obj_rot = transforms.matrix_to_quaternion(aug_obj_rot)
            all_seqs.append({
                "hand_pose_seq": ori_hand_seq,
                "obj_pose_seq": np.concatenate([ori_pos, aug_obj_rot.numpy()], axis=1),
                "conf": ori_confs,
                "obj_init_pos": ori_init_obj_pos,
            })


if __name__ == '__main__':
    meta_fn = os.path.join(data_root, 'meta.yaml')
    meta_info = yaml.safe_load(open(meta_fn, 'r'))
    obj_name = meta_info['obj_name']

    all_train_seqs = []
    all_test_seqs = []
    all_cmp_seqs = []

    total_train_frame_num = 0
    total_test_frame_num = 0

    for mi in tqdm.tqdm(meta_info['train']):
        seq_name = mi[0]
        start_idx = mi[1]
        end_idx = mi[2]
        data_prefix = os.path.join(data_root, seq_name)

        if ref_obj_fn is None:
            ref_obj_fn = os.path.join(data_prefix, 'obj_v.txt')
        obj2cano = LoadObj(os.path.join(data_prefix, 'obj_v.txt'),
                           os.path.join(data_prefix, 'obj_n.txt'))
        thetas, obj_pos, obj_rot, confs = LoadData(os.path.join(data_prefix, 'motion_seq.json'), obj2cano)
        thetas = thetas[start_idx: end_idx]
        obj_pos = obj_pos[start_idx: end_idx]
        obj_rot = obj_rot[start_idx: end_idx]
        confs = np.array(confs[start_idx: end_idx])
        hand_pose_seq, obj_pose_seq, obj_init_pos = load_motion(thetas, torch.cat([obj_pos, obj_rot], dim=1))
        all_train_seqs.append({
            "hand_pose_seq": hand_pose_seq.numpy(),
            "obj_pose_seq": obj_pose_seq.numpy(),
            "conf": confs,
            "obj_init_pos": obj_init_pos,
        })
        # symmetric object augmentation
        symmetric_aug(obj_name, obj_pose_seq.numpy(), hand_pose_seq.numpy(), confs, obj_init_pos, all_train_seqs)

        total_train_frame_num += len(thetas)

        if test_only:
            break

    for mi in tqdm.tqdm(meta_info['test']):
        seq_name = mi[0]
        start_idx = mi[1]
        end_idx = mi[2]
        data_prefix = os.path.join(data_root, seq_name)

        obj2cano = LoadObj(os.path.join(data_prefix, 'obj_v.txt'),
                           os.path.join(data_prefix, 'obj_n.txt'))
        thetas, obj_pos, obj_rot, confs = LoadData(os.path.join(data_prefix, 'motion_seq.json'), obj2cano)
        thetas = thetas[start_idx: end_idx]
        obj_pos = obj_pos[start_idx: end_idx]
        obj_rot = obj_rot[start_idx: end_idx]
        confs = np.array(confs[start_idx: end_idx])
        hand_pose_seq, obj_pose_seq, obj_init_pos = load_motion(thetas, torch.cat([obj_pos, obj_rot], dim=1))
        all_test_seqs.append({
            "hand_pose_seq": hand_pose_seq.numpy(),
            "obj_pose_seq": obj_pose_seq.numpy(),
            "conf": confs,
            "obj_init_pos": obj_init_pos,
        })

        total_test_frame_num += len(thetas)

    if meta_info.get('cmp') is not None:
        for mi in tqdm.tqdm(meta_info['cmp']):
            seq_name = mi[0]
            start_idx = mi[1]
            end_idx = mi[2]
            data_prefix = os.path.join(data_root, seq_name)

            obj2cano = LoadObj(os.path.join(data_prefix, 'obj_v.txt'),
                               os.path.join(data_prefix, 'obj_n.txt'))
            thetas, obj_pos, obj_rot, confs = LoadData(os.path.join(data_prefix, 'motion_seq.json'), obj2cano)
            thetas = thetas[start_idx: end_idx]
            obj_pos = obj_pos[start_idx: end_idx]
            obj_rot = obj_rot[start_idx: end_idx]
            confs = np.array(confs[start_idx: end_idx])
            hand_pose_seq, obj_pose_seq, obj_init_pos = load_motion(thetas, torch.cat([obj_pos, obj_rot], dim=1))
            all_cmp_seqs.append({
                "hand_pose_seq": hand_pose_seq.numpy(),
                "obj_pose_seq": obj_pose_seq.numpy(),
                "conf": confs,
                "obj_init_pos": obj_init_pos,
            })

    all_train_seqs.append(all_test_seqs[0])     # Add one validation seq

    print('Train seq num: ', len(meta_info['train']))
    print('Train frame num: ', total_train_frame_num)
    print('Test seq num: ', len(meta_info['test']))
    print('Test frame num: ', total_test_frame_num)

    if is_show_motion:
        hand_fn = 'assets/hand_model/spheremesh/sphere_mesh_hand.xml'
        obj_fn = meta_info['obj_model_fn']
        model_fn = 'dataset_model_tmp.xml'
        with open(model_fn, 'w') as f:
            model_xml = add_obj(hand_fn, obj_fn)
            f.write(model_xml)
        model = mjpy.load_model_from_path(model_fn)
        sim = mjpy.MjSim(model)
        viewer = mjpy.MjViewer(sim)
        ndof = all_train_seqs[0]['hand_pose_seq'].shape[1]

        for i, seq in enumerate(all_train_seqs[:]):
            hand_pose_seq = seq['hand_pose_seq']
            obj_pose_seq = seq['obj_pose_seq']

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

    if is_dump_motion:
        seq_name = obj_name + '_seq'
        train_fn_name = seq_name + '.pkl'
        test_fn_name = obj_name + '_test_seq.pkl'
        cmp_fn_name = obj_name + '_cmp_seq.pkl'
        if not test_only:
            pickle.dump({seq_name: all_train_seqs}, open(os.path.join(data_root, train_fn_name), 'wb'))
        pickle.dump({seq_name: all_test_seqs}, open(os.path.join(data_root, test_fn_name), 'wb'))
        pickle.dump({seq_name: all_cmp_seqs}, open(os.path.join(data_root, cmp_fn_name), 'wb'))
