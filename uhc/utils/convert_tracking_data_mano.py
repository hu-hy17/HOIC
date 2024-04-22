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

keypoint_id = [
    25,  # root
    19, 18, 17, 16,  # thumb (base -> end)
    11, 10, 9, 8,    # index
    15, 14, 13, 12,  # middle
    3, 2, 1, 0,       # ring
    7, 6, 5, 4,      # little
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

def LoadData(fn, o2c):
    json_root = json.load(open(fn, 'r'))
    joint_id = json_root['hand_info']['joint_id']

    motion_seq_json = json_root['motion_info']
    start_idx = motion_seq_json[0]['frame_id']
    seq_len = motion_seq_json[-1]['frame_id'] - start_idx + 1

    joint_pos_seq = []
    obj_pose_seq = []
    for i in range(seq_len):
        joint_pos = torch.Tensor(motion_seq_json[i]['hand_motion']).view(-1, 3)
        joint_pos_seq.append(joint_pos)
        obj_pose_seq.append(torch.Tensor(motion_seq_json[i]["obj_motion"]).view(4, 4))

    joint_pos_seq = torch.stack(joint_pos_seq, dim=0)
    joint_pos_seq = joint_pos_seq[:, joint_id, :]
    keypoint_seq = joint_pos_seq[:, keypoint_id, :]
    keypoint_seq[:, :, 2] *= -1

    obj_pose_seq = torch.stack(obj_pose_seq, dim=0)
    obj_pose_seq[:, 0:3, 3] /= 1000
    obj_pose_seq[:, 1:3, :] *= -1
    obj_pose_seq = torch.matmul(obj_pose_seq, torch.tensor(o2c).to(torch.float32))
    obj_pos_seq = obj_pose_seq[:, 0:3, 3]
    obj_rot_seq = transforms.matrix_to_quaternion(obj_pose_seq[:, 0:3, 0:3])

    return keypoint_seq / 1000, obj_pos_seq, obj_rot_seq


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
    obj2cano = obj_mesh.bounding_box_oriented.primitive.transform
    obj_mesh.apply_transform(np.linalg.inv(obj2cano))

    # obj_mesh.export('assets/tracking/Visual/box.stl')
    # obj_mesh.simplify_quadric_decimation(64)
    # obj_mesh.export('assets/tracking/Collision/box.stl')

    return obj2cano


def SolveManoParamsFromKp(keypoints):
    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True)
    th_v_shaped = torch.matmul(mano_layer.th_shapedirs,
                               mano_layer.th_betas.transpose(1, 0)).permute(2, 0, 1) + mano_layer.th_v_template
    th_j = torch.matmul(mano_layer.th_J_regressor, th_v_shaped)
    hand_root_pos = th_j[0][0]
    kp_nums = keypoints.shape[0]

    # torch optimization
    device = torch.device('cuda:0')
    keypoints_tensor = torch.Tensor(keypoints).to(device)
    mano_params = torch.zeros(kp_nums, 45 + 6).to(device)
    mano_params[:, 48:51] = keypoints_tensor[:, 0, :] - hand_root_pos.to(device)
    mano_params.requires_grad = True
    betas = torch.zeros(kp_nums, 10).to(device)
    mano_layer = mano_layer.to(device)
    max_epoch = 1000
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([mano_params], lr=5.0, momentum=0.9)

    for epoch in range(max_epoch):
        optimizer.zero_grad()
        _, pred_kp = mano_layer(th_pose_coeffs=mano_params[:, 0:48],
                                th_betas=betas,
                                th_trans=mano_params[:, 48:51])
        pred_kp = pred_kp / 1000
        loss = mse(pred_kp, keypoints_tensor) * mano_params.shape[0]
        loss.backward()
        optimizer.step()
        print('epoch %d, loss %f' % (epoch, loss.item() / mano_params.shape[0]))

    return mano_params.detach().cpu()


def trySolveManoParamsFromKp(keypoints):
    mano_layer = ManoLayer(mano_root='data/mano', use_pca=False, ncomps=45, flat_hand_mean=True)
    th_v_shaped = torch.matmul(mano_layer.th_shapedirs,
                               mano_layer.th_betas.transpose(1, 0)).permute(2, 0, 1) + mano_layer.th_v_template
    th_j = torch.matmul(mano_layer.th_J_regressor, th_v_shaped)
    hand_root_pos = th_j[0][0]
    kp = keypoints[0]
    device = torch.device('cpu')
    keypoints_tensor = torch.Tensor(kp).to(device).unsqueeze(0)
    mano_params = torch.zeros(1, 45 + 6).to(device)
    mano_params[:, 48:51] = keypoints_tensor[:, 0, :] - hand_root_pos
    mano_params.requires_grad = True
    betas = torch.zeros(1, 10).to(device)
    # betas.requires_grad = True
    mano_layer = mano_layer.to(device)
    max_epoch = 1000
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([mano_params, betas], lr=50.0, momentum=0.9)

    vis_optim(kp, mano_layer, mano_params)

    for epoch in range(max_epoch):
        optimizer.zero_grad()
        _, pred_kp = mano_layer(th_pose_coeffs=mano_params[:, 0:48],
                                th_betas=betas,
                                th_trans=mano_params[:, 48:51])
        pred_kp = pred_kp / 1000
        loss = mse(pred_kp, keypoints_tensor) + 0.0001 * torch.norm(betas)
        loss.backward()
        optimizer.step()
        print('epoch %d, loss %f' % (epoch, loss.item()))

    vis_optim(kp, mano_layer, mano_params)


def vis_optim(kps, mano_layer, mano_params):
    import open3d as o3d

    # generate point for each keypoints
    kp_num = kps.shape[0]
    kp_mesh = o3d.geometry.TriangleMesh()
    for i in range(0, kp_num):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.translate(kps[i, :])
        mesh_sphere.paint_uniform_color(kp_colors[i])
        kp_mesh += mesh_sphere

    hand_v, hand_j = mano_layer(th_pose_coeffs=mano_params[:, 0:48],
                                th_betas=torch.zeros((1, 10)),
                                th_trans=mano_params[:, 48:51])
    hand_v = hand_v / 1000
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_v[0].detach().cpu().numpy())
    hand_mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces.detach().cpu().numpy())
    hand_mesh.compute_vertex_normals()

    o3d.visualization.draw([hand_mesh, kp_mesh])


def load_motion(hand_pose_seq, obj_pose_seq):
    # transfer hand pca pose to dofs
    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True)
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

    obj_rot_mat = transforms.quaternion_to_matrix(obj_pose_seq[:, 3:7])
    obj_pos_seq = obj_pose_seq[:, 0:3].clone()
    hand_root_rot_mat = transforms.axis_angle_to_matrix(hand_pose_seq[:, :3])
    hand_root_pos = hand_pose_seq[:, 48:51].clone()

    obj_rot_mat = obj_rot_mat[:, [0, 2, 1], :]
    obj_rot_mat[:, 1, :] *= -1
    hand_root_rot_mat = hand_root_rot_mat[:, [0, 2, 1], :]
    hand_root_rot_mat[:, 1, :] *= -1
    obj_pos_seq = obj_pos_seq[:, [0, 2, 1]]
    obj_pos_seq[:, 1] *= -1
    hand_root_pos = hand_root_pos[:, [0, 2, 1]]
    hand_root_pos[:, 1] *= -1

    # Swap Y and Z axis
    obj_pose_seq[:, 0:3] = obj_pos_seq
    obj_pose_seq[:, 3:7] = transforms.matrix_to_quaternion(obj_rot_mat)
    hand_pose_seq[:, :3] = transforms.matrix_to_axis_angle(hand_root_rot_mat)
    hand_pose_seq[:, 48:51] = hand_root_pos

    # Translate hand and object
    obj_init_pos = obj_pose_seq[0, 0:3].clone()
    obj_pose_seq[:, 0:3] = obj_pose_seq[:, 0:3] - obj_init_pos
    hand_pose_seq[:, 48:51] = hand_pose_seq[:, 48:51] - obj_init_pos

    # z-axis offset
    z_offset = 0.7
    obj_pose_seq[:, 2] += z_offset
    hand_pose_seq[:, 50] += z_offset

    # align coordinate
    # z_offset = 0.5 + 0.2  # 0.0683-bottle 0.025-airplane
    # obj_rot_mat = transforms.quaternion_to_matrix(obj_pose_seq[:, 3:7])
    # obj_init_rot_mat = obj_rot_mat[0].clone()
    # obj_init_pos = obj_pose_seq[0, 0:3].clone()
    # obj_init_rot_inv = obj_init_rot_mat.transpose(0, 1)
    #
    # # align object
    # obj_rot_mat = torch.matmul(obj_init_rot_inv, obj_rot_mat)
    # obj_pose_seq[:, 3:7] = transforms.matrix_to_quaternion(obj_rot_mat)
    # obj_pose_seq[:, 0:3] = (obj_pose_seq[:, 0:3] - obj_init_pos).mm(obj_init_rot_mat)
    #
    # # align hand
    # hand_pose_seq[:, 48:] = (hand_pose_seq[:, 48:] - obj_init_pos).mm(obj_init_rot_mat)
    # hand_global_rot = hand_pose_seq[:, :3].clone()
    # hand_global_rot = torch.matmul(obj_init_rot_inv, transforms.axis_angle_to_matrix(hand_global_rot))
    # hand_global_rot = transforms.matrix_to_axis_angle(hand_global_rot)
    # hand_pose_seq[:, :3] = hand_global_rot
    #
    # # z-axis offset
    # obj_pose_seq[:, 2] += z_offset
    # hand_pose_seq[:, 50] += z_offset

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

    return hand_pose_seq[:], obj_pose_seq[:], hand_j / 1000


def add_obj(hand_fn, obj_fn):
    if obj_fn is None:
        raise ValueError("obj_fn is None!")
    mj_xml = MujocoXML(hand_fn)
    mj_xml.merge(MujocoXML(obj_fn))
    mj_xml.get_xml()
    return mj_xml.get_xml()


def show_motion(hand_pose_seq, obj_pose_seq, save_root=None):
    hand_fn = 'assets/hand_model/manohand_reduce2/mano_hand.xml'
    obj_fn = 'assets/SingleDepth/box.xml'
    model_fn = 'dataset_model_tmp2.xml'
    with open(model_fn, 'w') as f:
        model_xml = add_obj(hand_fn, obj_fn)
        f.write(model_xml)
    model = mjpy.load_model_from_path(model_fn)
    # model = mjpy.load_model_from_xml(add_obj(hand_fn, obj_fn))
    sim = mjpy.MjSim(model)
    viewer = mjpy.MjViewer(sim)
    ndof = hand_pose_seq.shape[1]

    t = 0
    while True:
        print(str(t) + '/' + str(hand_pose_seq.shape[0]))
        sim.data.qpos[:ndof] = hand_pose_seq[t, :]
        sim.data.qpos[ndof:] = obj_pose_seq[t, :]
        sim.forward()
        viewer.render()
        # if save_root is not None:
        #     img = sim.render(640, 480, device_id=0, depth=False)
        #     cv2.imwrite(save_root + '/' + str(t) + '.png', img)
        t += 1
        if t >= hand_pose_seq.shape[0]:
            break
        time.sleep(0.01)


if __name__ == '__main__':
    select_seq_seg_idx = [
        [[0, 1150], [1250, 2200]],
        [[0, 700], [750, 1400]],
        [[0, 850]],
        [[0, 1400]],
        [[0, 750]],
        [[0, 900]],
        [[0, 950], [1000, 1300]],
        [[0, 1200]],
    ]

    data_prefix_arr = []
    data_root = 'sample_data/SingleDepth/Box2/'
    for i in range(1, 9):
        data_prefix_arr.append(data_root + 'Seq' + str(i) + '/')

    solve_mano_params = True
    is_show_motion = False
    is_dump_motion = True

    keypoints_arr = []
    obj_pos_arr = []
    obj_rot_arr = []
    seq_seg_idx = [0]

    for data_prefix in data_prefix_arr:
        obj2cano = LoadObj(os.path.join(data_prefix, 'obj_v.txt'),
                           os.path.join(data_prefix, 'obj_n.txt'))
        keypoints, obj_pos, obj_rot = LoadData(os.path.join(data_prefix, 'motion_seq.json'), obj2cano)
        keypoints_arr.append(keypoints)
        obj_pos_arr.append(obj_pos)
        obj_rot_arr.append(obj_rot)
        seq_seg_idx.append(seq_seg_idx[-1] + keypoints.shape[0])

    # testSolveManoParamsFromKp(keypoints)
    if solve_mano_params:
        mano_params = SolveManoParamsFromKp(torch.cat(keypoints_arr, dim=0))
    all_seqs = []

    for i in range(len(data_prefix_arr)):
        data_prefix = data_prefix_arr[i]
        obj_pos = obj_pos_arr[i]
        obj_rot = obj_rot_arr[i]
        if solve_mano_params:
            mano_p = mano_params[seq_seg_idx[i]: seq_seg_idx[i + 1]]
            pickle.dump(mano_p, open(os.path.join(data_prefix, 'mano_params.pkl'), 'wb'))
        mano_p = pickle.load(open(os.path.join(data_prefix, 'mano_params.pkl'), 'rb'))
        hand_pose_seq, obj_pose_seq, _ = load_motion(mano_p, torch.cat([obj_pos, obj_rot], dim=1))
        hand_pose_seq = hand_pose_seq[:, hand_dof_reduce_map]
        for sq in select_seq_seg_idx[i]:
            all_seqs.append({
                "hand_pose_seq": hand_pose_seq[sq[0]: sq[1]].numpy(),
                "obj_pose_seq": obj_pose_seq[sq[0]: sq[1]].numpy()
            })

    seq_name = 'box_seq'

    if is_show_motion:
        for i, seq in enumerate(all_seqs):
            show_motion(seq['hand_pose_seq'], seq['obj_pose_seq'], 'out/Render/' + seq_name + '_' + str(i))

    if is_dump_motion:
        pickle.dump({seq_name: all_seqs}, open(os.path.join(data_root, seq_name + '_mano' + '.pkl'), 'wb'))
