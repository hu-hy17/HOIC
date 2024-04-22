import os
import time
from collections import defaultdict

import cv2
import numpy as np
import trimesh
import torch
import open3d as o3d
import pyrender
import transforms3d as t3d
import matplotlib.pyplot as plt

from manopth.manolayer import ManoLayer

keypoint_id = [
    0,  # root
    17, 18, 19, 20,  # thumb (base -> end)
    1, 2, 3, 4,  # index
    5, 6, 7, 8,  # middle
    13, 14, 15, 16,  # ring
    9, 10, 11, 12,  # little
]

rad = 0.001


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def create_arrow(start_pos, force, r=0.001):
    end_pos = start_pos + force / 100
    cone_trans = np.eye(4)
    cone_trans[0:3, 3] = end_pos
    cone_trans[0:3, 0:3] = rotation_matrix_from_vectors(np.array([0, 0, 1]), force)
    cy_mesh = trimesh.creation.cylinder(r, segment=np.stack([start_pos, end_pos]))
    cone_mesh = trimesh.creation.cone(2 * r, height=0.01, transform=cone_trans)
    return cy_mesh + cone_mesh


class ForceVisualizer:
    def __init__(self, obj_mesh_fn, mano_path, root_offset, add_marker=True, save=False, save_dir=None):
        self.obj_mesh = trimesh.load_mesh(obj_mesh_fn)
        self.mano_path = mano_path
        self.mano_layer = ManoLayer(mano_root=mano_path, use_pca=False, ncomps=45, flat_hand_mean=True)
        self.root_offset = root_offset
        self.add_marker = add_marker
        # 骨骼连接
        self.bone_connections = [
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20]
        ]
        # 设置每个手指的颜色
        self.finger_colors = np.array([
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 0.7, 0.3],  # 橙色
            [1.0, 0.0, 1.0],  # 洋红
        ])
        self.point_color_idx = np.array([0,
                                         0, 0, 0, 0,
                                         1, 1, 1, 1,
                                         2, 2, 2, 2,
                                         3, 3, 3, 3,
                                         4, 4, 4, 4])
        self.line_color_idx = np.array([0, 1, 2, 3, 4,
                                        0, 0, 0,
                                        1, 1, 1,
                                        2, 2, 2,
                                        3, 3, 3,
                                        4, 4, 4])

    def update_skeleton(self, scene, kps, opose):
        for node in scene.get_nodes():
            if node.name is None:
                continue
            elif node.name.startswith('kp') or node.name.startswith('bone') or node.name.startswith('arrow'):
                scene.remove_node(node)
        # for i, point in enumerate(kps):
        #     mesh = pyrender.Mesh.from_points(np.array([point]), colors=self.finger_colors[self.point_color_idx[i]])
        #     scene.add(mesh, 'kp' + str(i))
        for i, connection in enumerate(self.bone_connections):
            color = self.finger_colors[self.line_color_idx[i]]
            mat = pyrender.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[color[0], color[1], color[2], 1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            line = pyrender.Mesh.from_trimesh(
                trimesh.creation.cylinder(0.0025, segment=np.stack([kps[connection[0]], kps[connection[1]]])), material=mat)
            scene.add(line, 'bone' + str(i))

        arrow_color = [
            [1.0, 0, 0, 1.0],
            [0, 1.0, 0, 1.0],
            [0, 0, 1.0, 1.0]
        ]

        for i in range(3):
            pos = opose[:3, 3]
            dir = opose[:3, i] * 3
            mat = pyrender.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=arrow_color[i],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            mesh = pyrender.Mesh.from_trimesh(create_arrow(pos, dir, r=0.0025), material=mat)
            scene.add(mesh, 'arrow' + str(i))

    def classify_cp(self, cps, cpf, cpb):
        frame_num = len(cps) // 15
        c_cps = []
        c_cpf = []
        for i in range(frame_num):
            cps_map = defaultdict(list)
            cpf_map = defaultdict(list)
            for j in range(15):
                idx = 15 * i + j
                c_num = cps[idx].shape[0]
                for k in range(c_num):
                    body_idx = cpb[idx][k]
                    c_pos = cps[idx][k][0:3]
                    c_f = np.matmul(cps[idx][k][3:12].reshape(3, 3).T, cpf[idx][k][0:3])
                    cps_map[body_idx].append(c_pos)
                    cpf_map[body_idx].append(c_f)
            avg_cps = []
            avg_cpf = []
            for key in cps_map.keys():
                avg_cp = np.mean(np.array(cps_map[key]), axis=0)
                avg_cf = np.mean(np.array(cpf_map[key]), axis=0)
                avg_cps.append(avg_cp)
                avg_cpf.append(avg_cf)
            c_cps.append(np.array(avg_cps))
            c_cpf.append(np.array(avg_cpf))

        return c_cps, c_cpf

    def render_force(self, h_kps, obj_pose, cps, cpf, cpb, cps_ft, rfc_rwd=None, save=False, save_dir=None):
        # Reduce motion frame rate
        frame_num = h_kps.shape[0]
        h_kps = h_kps[np.arange(0, frame_num, 15)]
        obj_pose = obj_pose[np.arange(0, frame_num, 15)]
        cps_ft = np.array(cps_ft)[np.arange(0, frame_num, 15)]
        cps, cpf = self.classify_cp(cps, cpf, cpb)
        frame_num = h_kps.shape[0]

        h_kps[:, :, 2] -= 0.7
        mano_params, betas = self.solve_mano_params_from_kp(h_kps)

        # Get hand mesh
        device = torch.device('cuda:0')
        mano_layer = self.mano_layer.to(device)
        h_mesh_v_arr, _ = mano_layer(th_pose_coeffs=mano_params[:, 0:48],
                                     th_betas=betas,
                                     th_trans=mano_params[:, 48:51])
        h_mesh_v_arr /= 1000
        h_mesh_f = mano_layer.th_faces

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
        o_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.8,
            smooth=False
        )
        h_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[1.0, 0.796, 0.643, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        arrow_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.0, 0.0, 1.0, 1.0],
            metallicFactor=0.5,
            roughnessFactor=0.5
        )
        arrow_mat2 = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.0, 1.0, 0.0, 1.0],
            metallicFactor=0.5,
            roughnessFactor=0.5
        )
        arrow_mat3 = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.0, 1.0, 1.0, 1.0],
            metallicFactor=0.5,
            roughnessFactor=0.5
        )
        cp_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[1.0, 0.0, 0.0, 1.0],
            metallicFactor=0.5,
            roughnessFactor=0.5
        )
        o_mesh = pyrender.Mesh.from_trimesh(self.obj_mesh, material=o_mat)
        h_mesh = trimesh.Trimesh(vertices=h_mesh_v_arr[0].detach().cpu().numpy(),
                                 faces=h_mesh_f.cpu().numpy())
        h_mesh = pyrender.Mesh.from_trimesh(h_mesh, material=h_mat)
        scene.add(o_mesh, name='obj')
        scene.add(h_mesh, name='hand')
        camera = pyrender.PerspectiveCamera(yfov=np.pi * 0.2333333, aspectRatio=1.333333)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, -0.0257],
            [0.0, 0.0, -1.0, -0.004077],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        camera_pose[0:3, 3] -= self.root_offset
        scene.add(camera, pose=camera_pose)
        # light = pyrender.SpotLight(color=np.ones(3), intensity=0.5,
        #                            innerConeAngle=np.pi / 16.0,
        #                            outerConeAngle=np.pi / 6.0)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        paused = False

        def pause_render(viewer):
            nonlocal paused
            paused = not paused

        if save:
            r = pyrender.OffscreenRenderer(640, 480)
        else:
            r = pyrender.Viewer(scene,
                                viewport_size=(1920, 1440),
                                run_in_thread=True,
                                registered_keys={' ': pause_render},
                                use_raymond_lighting=True)

        trimesh.creation.axis()

        frame_idx = 0
        while frame_idx < frame_num:
            if not save and paused:
                time.sleep(0.01)
                continue
            h_mesh = trimesh.Trimesh(vertices=h_mesh_v_arr[frame_idx].detach().cpu().numpy(),
                                     faces=h_mesh_f.cpu().numpy())
            h_mesh = pyrender.Mesh.from_trimesh(h_mesh, material=h_mat)

            # obj transformation
            o_trans = np.eye(4)
            o_trans[0:3, 3] = obj_pose[frame_idx, 0:3] - np.array([0, 0, 0.7])
            o_trans[0:3, 0:3] = t3d.quaternions.quat2mat(obj_pose[frame_idx, 3:])

            if not save:
                r.render_lock.acquire()
            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if node.name == 'obj':
                    scene.set_pose(node, o_trans)
                elif node.name == 'hand':
                    scene.remove_node(node)
                elif node.name.startswith('arrow'):
                    scene.remove_node(node)
                elif node.name.startswith('marker'):
                    scene.remove_node(node)
            scene.add(h_mesh, name='hand')

            if self.add_marker:
                # Add force
                for c_idx in range(cps[frame_idx].shape[0]):
                    pos = cps[frame_idx][c_idx] - np.array([0, 0, 0.7])
                    force = cpf[frame_idx][c_idx]
                    arrow_mesh = create_arrow(pos, force)
                    arrow_mesh = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_mat)
                    ball_mesh = trimesh.primitives.Sphere(radius=4 * rad, center=pos)
                    ball_mesh = pyrender.Mesh.from_trimesh(ball_mesh, material=cp_mat)
                    scene.add(arrow_mesh, name='arrow' + str(c_idx))
                    scene.add(ball_mesh, name='arrow' + str(c_idx) + '-s')

                # Add compensate force
                pos = obj_pose[frame_idx, 0:3] - np.array([0, 0, 0.7])
                arrow_mesh = create_arrow(pos, cps_ft[frame_idx][:3])
                arrow_mesh = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_mat2)
                ball_mesh = trimesh.primitives.Sphere(radius=4 * rad, center=pos)
                ball_mesh = pyrender.Mesh.from_trimesh(ball_mesh, material=cp_mat)
                scene.add(arrow_mesh, name='arrow' + '-vf')
                scene.add(ball_mesh, name='arrow' + '-vfs')
                arrow_mesh = create_arrow(pos, cps_ft[frame_idx][3:6])
                arrow_mesh = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_mat3)
                scene.add(arrow_mesh, name='arrow' + '-vt')

                # Add rfc reward marker
                marker = trimesh.primitives.Cylinder(radius=0.01, height=rfc_rwd[frame_idx] / 20)
                marker = marker.apply_translation(np.array([0.2, 0, rfc_rwd[frame_idx] / 40]))
                marker = pyrender.Mesh.from_trimesh(marker)
                scene.add(marker, name='marker')

            if not save:
                r.render_lock.release()
            else:
                color, _ = r.render(scene)
                cv2.imwrite(save_dir + '/render_%04d.png' % frame_idx, color[:, :, [2, 1, 0]])

            frame_idx += 1

    def render_kin(self, h_kps, obj_pose, save=False, save_dir=None):
        frame_num = h_kps.shape[0]

        h_kps[:, :, 2] -= 0.7
        mano_params, betas = self.solve_mano_params_from_kp(h_kps)

        # Get hand mesh
        device = torch.device('cuda:0')
        mano_layer = self.mano_layer.to(device)
        h_mesh_v_arr, _ = mano_layer(th_pose_coeffs=mano_params[:, 0:48],
                                     th_betas=betas,
                                     th_trans=mano_params[:, 48:51])
        h_mesh_v_arr /= 1000
        h_mesh_f = mano_layer.th_faces

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
        o_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.2,
            roughnessFactor=0.8,
            smooth=False
        )
        h_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[1.0, 0.796, 0.643, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )

        o_mesh = pyrender.Mesh.from_trimesh(self.obj_mesh, material=o_mat)
        h_mesh = trimesh.Trimesh(vertices=h_mesh_v_arr[0].detach().cpu().numpy(),
                                 faces=h_mesh_f.cpu().numpy())
        h_mesh = pyrender.Mesh.from_trimesh(h_mesh, material=h_mat)
        scene.add(o_mesh, name='obj')
        scene.add(h_mesh, name='hand')
        camera = pyrender.PerspectiveCamera(yfov=np.pi * 0.2333333, aspectRatio=1.333333)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, -0.0257],
            [0.0, 0.0, -1.0, -0.004077],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        camera_pose[0:3, 3] -= self.root_offset
        scene.add(camera, pose=camera_pose)
        # light = pyrender.SpotLight(color=np.ones(3), intensity=0.5,
        #                            innerConeAngle=np.pi / 16.0,
        #                            outerConeAngle=np.pi / 6.0)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        paused = False

        def pause_render(viewer):
            nonlocal paused
            paused = not paused

        if save:
            r = pyrender.OffscreenRenderer(640, 480)
        else:
            r = pyrender.Viewer(scene,
                                viewport_size=(1920, 1440),
                                run_in_thread=True,
                                registered_keys={' ': pause_render},
                                use_raymond_lighting=True)

        trimesh.creation.axis()

        frame_idx = 0
        while frame_idx < frame_num:
            if not save and paused:
                time.sleep(0.01)
                continue
            h_mesh = trimesh.Trimesh(vertices=h_mesh_v_arr[frame_idx].detach().cpu().numpy(),
                                     faces=h_mesh_f.cpu().numpy())
            h_mesh = pyrender.Mesh.from_trimesh(h_mesh, material=h_mat)

            # obj transformation
            o_trans = np.eye(4)
            o_trans[0:3, 3] = obj_pose[frame_idx, 0:3] - np.array([0, 0, 0.7])
            o_trans[0:3, 0:3] = t3d.quaternions.quat2mat(obj_pose[frame_idx, 3:])

            if not save:
                r.render_lock.acquire()
            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if node.name == 'obj':
                    scene.set_pose(node, o_trans)
                elif node.name == 'hand':
                    scene.remove_node(node)
            scene.add(h_mesh, name='hand')

            if not save:
                r.render_lock.release()
            else:
                color, _ = r.render(scene)
                cv2.imwrite(os.path.join(save_dir + '/render_%04d.png' % frame_idx), color[:, :, [2, 1, 0]])

            frame_idx += 1

    def render_skeleton(self, h_kps, obj_pose, save=False, save_dir=None):
        frame_num = h_kps.shape[0]

        h_kps[:, :, 2] -= 0.7

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
        o_mat = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.8, 0.8, 0.8, 0.8],
            metallicFactor=0.2,
            roughnessFactor=0.8,
            smooth=False
        )

        o_mesh = pyrender.Mesh.from_trimesh(self.obj_mesh, material=o_mat)
        scene.add(o_mesh, name='obj')
        camera = pyrender.PerspectiveCamera(yfov=np.pi * 0.2333333, aspectRatio=1.333333)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, -0.0257],
            [0.0, 0.0, -1.0, -0.004077],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        camera_pose[0:3, 3] -= self.root_offset
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        paused = False

        def pause_render(viewer):
            nonlocal paused
            paused = not paused

        if save:
            r = pyrender.OffscreenRenderer(640, 480)
        else:
            r = pyrender.Viewer(scene,
                                viewport_size=(1920, 1440),
                                run_in_thread=True,
                                registered_keys={' ': pause_render},
                                use_raymond_lighting=True)

        trimesh.creation.axis()

        frame_idx = 0
        while frame_idx < frame_num:
            if not save and paused:
                time.sleep(0.01)
                continue

            # obj transformation
            o_trans = np.eye(4)
            o_trans[0:3, 3] = obj_pose[frame_idx, 0:3] - np.array([0, 0, 0.7])
            o_trans[0:3, 0:3] = t3d.quaternions.quat2mat(obj_pose[frame_idx, 3:])

            if not save:
                r.render_lock.acquire()

            self.update_skeleton(scene, h_kps[frame_idx], o_trans)
            for node in scene.get_nodes():
                if node.name is None:
                    continue
                if node.name == 'obj':
                    scene.set_pose(node, o_trans)

            if not save:
                r.render_lock.release()
            else:
                color, _ = r.render(scene)
                cv2.imwrite(os.path.join(save_dir + '/render_%04d.png' % frame_idx), color[:, :, [2, 1, 0]])

            frame_idx += 1

    def solve_mano_params_from_kp(self, keypoints):
        mano_layer = self.mano_layer
        mano_layer.zero_grad()
        th_v_shaped = torch.matmul(mano_layer.th_shapedirs,
                                   mano_layer.th_betas.transpose(1, 0)).permute(2, 0, 1) + mano_layer.th_v_template
        th_j = torch.matmul(mano_layer.th_J_regressor, th_v_shaped)
        hand_root_pos = th_j[0][0]
        kp_nums = keypoints.shape[0]

        # torch optimization
        device = torch.device('cuda:0')
        keypoints_tensor = torch.Tensor(keypoints).to(device)
        keypoints_tensor = keypoints_tensor[:, keypoint_id]
        mano_params = torch.zeros(kp_nums, 45 + 6).to(device)
        mano_params[:, 48:51] = keypoints_tensor[:, 0, :] - hand_root_pos.to(device)
        mano_params.requires_grad = True
        betas = torch.Tensor([-2.61435056, -1.16743336, -2.80988378, 0.12670897, -0.08323125, 2.28185672,
                              -0.05833138, -2.95105206, -3.43976417, 0.30667237])
        betas = torch.tile(betas, (kp_nums, 1)).to(device)
        mano_layer = mano_layer.to(device)
        max_epoch = 1000
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.SGD([mano_params, betas], lr=5, momentum=0.9)

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

        return mano_params.detach(), betas.detach()
