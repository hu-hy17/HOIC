import glob
import json

import cv2
import cvxopt
import mujoco_py as mjpy
import numpy as np
import qpsolvers
import torch
import transforms3d as t3d
import trimesh

from pysdf import SDF
from uhc.utils.transforms import quaternion_to_matrix, matrix_to_axis_angle
from uhc.utils.merge_model import merge_model

class PhysMetrics:
    def __init__(self, model, qpos_seq, obj_mesh_fn):
        self.model = model
        self.qpos_seq = qpos_seq
        self.sim = mjpy.MjSim(model)

        # prepare contact computation
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
        self.hand_geom_num = self.hand_geom_range[1] - self.hand_geom_range[0] + 1
        self.obj_geom_range = [-1, -1]
        self.obj_geom_range[0] = self.hand_geom_range[1] + 1
        for i in range(self.obj_geom_range[0], len(self.model.geom_names)):
            name = self.model.geom_names[i]
            if not name.startswith('C_'):
                self.obj_geom_range[1] = i - 1
                break

        self.box_size = [0.0165, 0.0265, 0.049]
        self.obj_mass = model.body_mass[-1]
        self.obj_inertia = model.body_inertia[-1]

        # load object mesh for penetration computation
        obj_mesh = trimesh.load_mesh(obj_mesh_fn)
        self.obj_sdf = SDF(obj_mesh.vertices, obj_mesh.faces)

    def eval_penetration(self):
        pene_arr = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:] = qpos
            self.sim.forward()
            query_points = []
            for contact in self.sim.data.contact[:self.sim.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                        self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                    cp_pos_global = contact.pos.copy()
                    obj_pos = qpos[-7: -4]
                    obj_quat = qpos[-4:]
                    rot_mat = t3d.quaternions.quat2mat(obj_quat)
                    cp_pos_local = np.matmul(np.linalg.inv(rot_mat), cp_pos_global - obj_pos)
                    query_points.append(cp_pos_local)
            if len(query_points) > 0:
                query_points = np.stack(query_points)
                all_sdf = self.obj_sdf(query_points)
                all_sdf[np.where(all_sdf < 0)] = 0
                pene_arr.append(np.mean(all_sdf) * 1000)
            else:
                pene_arr.append(0)
        return pene_arr

    def eval_penetration2(self):
        pene_arr = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:] = qpos
            self.sim.forward()
            pene_sum = 0
            for contact in self.sim.data.contact[:self.sim.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                        self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                    cp_pos_global = contact.pos.copy()
                    obj_pos = qpos[-7: -4]
                    obj_quat = qpos[-4:]
                    rot_mat = t3d.quaternions.quat2mat(obj_quat)
                    cp_pos_local = np.matmul(np.linalg.inv(rot_mat), cp_pos_global - obj_pos)
                    pene_depth = 100
                    for ai in range(3):
                        if cp_pos_local[ai] >= self.box_size[ai] or cp_pos_local[ai] <= -self.box_size[ai]:
                            pene_depth = 0
                            break
                        axis_pene = self.box_size[ai] - cp_pos_local[ai] if cp_pos_local[ai] >= 0 else cp_pos_local[ai] + self.box_size[ai]
                        pene_depth = min(pene_depth, axis_pene)
                    pene_sum += pene_depth * 2
            pene_arr.append(pene_sum * 1000 / self.sim.data.ncon)

        return pene_arr

    def eval_jitter(self):
        # Object jitter
        obj_pos_seq = self.qpos_seq[:, -7:-4]
        obj_quat_seq = self.qpos_seq[:, -4:]

        # compute object velocity and acc
        obj_vel_seq = np.gradient(obj_pos_seq, axis=0) * 30
        obj_acc_seq = np.gradient(obj_vel_seq, axis=0) * 30

        # compute object angle velocity and acc
        obj_angle_vel_seq = np.zeros_like(obj_quat_seq[:, 0:3])
        rot_mat_seq = quaternion_to_matrix(torch.Tensor(obj_quat_seq))
        relative_rot_seq = torch.matmul(rot_mat_seq[1:], torch.transpose(rot_mat_seq[:-1], 1, 2))
        obj_angle_vel_seq[1:] = matrix_to_axis_angle(relative_rot_seq).numpy() * 30
        obj_angle_vel_seq[0] = obj_angle_vel_seq[1]
        obj_angle_acc_seq = np.gradient(obj_angle_vel_seq, axis=0) * 30
        obj_angle_vel_seq = torch.Tensor(obj_angle_vel_seq[:, :, np.newaxis])
        obj_angle_acc_seq = torch.Tensor(obj_angle_acc_seq[:, :, np.newaxis])
        obj_angle_acc_seq = torch.squeeze(obj_angle_acc_seq).numpy()

        obj_avg_acc = np.mean(np.linalg.norm(obj_acc_seq, axis=-1))
        obj_avg_angle_acc = np.mean(np.linalg.norm(obj_angle_acc_seq, axis=-1))

        # Hand keypoints jitter
        joint_pos_seq = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:] = qpos
            self.sim.forward()
            joint_pos = self.sim.data.body_xpos[self.hand_body_idx]
            joint_pos_seq.append(joint_pos)
        joint_pos_seq = np.stack(joint_pos_seq)
        joint_vel_seq = np.gradient(joint_pos_seq, axis=0) * 30
        joint_acc_seq = np.gradient(joint_vel_seq, axis=0) * 30
        hand_avg_acc = np.linalg.norm(joint_acc_seq, axis=-1).mean()

        return hand_avg_acc, obj_avg_acc, obj_avg_angle_acc

    def eval_contact_point(self):
        cp_num_arr = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:] = qpos
            self.sim.forward()
            cp_num = 0
            for contact in self.sim.data.contact[:self.sim.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                        self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                    cp_num += 1
            cp_num_arr.append(cp_num)
        return cp_num_arr

    def solve_force(self, target_force, target_torque, obj_contacts, obj_center):
        n_c = len(obj_contacts)
        if n_c == 0:
            return np.linalg.norm(target_force) + np.linalg.norm(target_torque)
        friction_coef = 1.0
        A_arr = []
        rA_arr = []
        dx = 0.0025
        for i in range(0, n_c):
            pos = obj_contacts[i, :3]
            frame = obj_contacts[i, 3:12].reshape(3, 3)
            delta_pos = np.array([np.zeros(3), frame[1], -frame[1], frame[2], -frame[2]]) * dx
            for j in range(5):
                e_pos = pos + delta_pos[j]
                # convert to polynomial friction cone
                x1 = (frame[0] + friction_coef * frame[1]) / np.sqrt(1 + friction_coef ** 2)
                x2 = (frame[0] - friction_coef * frame[1]) / np.sqrt(1 + friction_coef ** 2)
                x3 = (frame[0] + friction_coef * frame[2]) / np.sqrt(1 + friction_coef ** 2)
                x4 = (frame[0] - friction_coef * frame[2]) / np.sqrt(1 + friction_coef ** 2)
                A = np.stack([x1, x2, x3, x4]).T
                A_arr.append(A)

                r = e_pos - obj_center
                r_x = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                rA_arr.append(np.matmul(r_x, A))

        J_f = np.concatenate(A_arr, axis=1)
        J_t = np.concatenate(rA_arr, axis=1)

        # Solve QP
        w_t = 1
        Q = 2 * (np.matmul(J_f.T, J_f) + w_t * np.matmul(J_t.T, J_t)) + 1e-7 * np.eye(4 * n_c * 5)
        p = -2 * np.matmul(J_f.T, target_force) - 2 * w_t * np.matmul(J_t.T, target_torque)
        G = -np.eye(4 * n_c * 5)
        h = np.zeros(4 * n_c * 5)
        res = qpsolvers.solve_qp(Q, p, G, h, solver='daqp')

        if res is not None:
            rest_ft = np.linalg.norm(J_f @ res - target_force) + w_t * np.linalg.norm(J_t @ res - target_torque)
        else:
            rest_ft = np.linalg.norm(target_force) + w_t * np.linalg.norm(target_torque)

        return rest_ft

    def obtain_target_ft(self):
        obj_pos_seq = self.qpos_seq[:, -7:-4]
        obj_quat_seq = self.qpos_seq[:, -4:]

        # compute object velocity and acc
        obj_vel_seq = np.gradient(obj_pos_seq, axis=0) * 30
        obj_acc_seq = np.gradient(obj_vel_seq, axis=0) * 30

        # compute object angle velocity and acc
        obj_angle_vel_seq = np.zeros_like(obj_quat_seq[:, 0:3])
        rot_mat_seq = quaternion_to_matrix(torch.Tensor(obj_quat_seq))
        relative_rot_seq = torch.matmul(rot_mat_seq[1:], torch.transpose(rot_mat_seq[:-1], 1, 2))
        obj_angle_vel_seq[1:] = matrix_to_axis_angle(relative_rot_seq).numpy() * 30
        obj_angle_vel_seq[0] = obj_angle_vel_seq[1]
        obj_angle_acc_seq = np.gradient(obj_angle_vel_seq, axis=0) * 30
        obj_angle_vel_seq = torch.Tensor(obj_angle_vel_seq[:, :, np.newaxis])
        obj_angle_acc_seq = torch.Tensor(obj_angle_acc_seq[:, :, np.newaxis])

        target_force_seq = self.obj_mass * (obj_acc_seq + np.array([0, 0, 9.8]))

        Ib = torch.diag(torch.Tensor(self.obj_inertia))
        Is_seq = torch.bmm(torch.tile(Ib, (len(rot_mat_seq), 1, 1)), torch.transpose(rot_mat_seq, 1, 2))
        Is_seq = torch.bmm(rot_mat_seq, Is_seq)
        target_torque_seq = torch.bmm(Is_seq, obj_angle_acc_seq) + torch.cross(obj_angle_vel_seq, torch.bmm(Is_seq, obj_angle_vel_seq))
        target_torque_seq = target_torque_seq.squeeze().numpy()

        return target_force_seq, target_torque_seq

    def eval_stable(self):
        target_force_seq, target_torque_seq = self.obtain_target_ft()
        rest_ft_arr = []
        for i, qpos in enumerate(self.qpos_seq):
            self.sim.data.qpos[:] = qpos
            self.sim.forward()
            obj_contacts = []
            for contact in self.sim.data.contact[:self.sim.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if self.hand_geom_range[0] <= g1 <= self.hand_geom_range[1] and \
                        self.obj_geom_range[0] <= g2 <= self.obj_geom_range[1]:
                    obj_contacts.append(np.concatenate([contact.pos.copy(), contact.frame.copy()]))
            target_force = target_force_seq[i]
            target_torque = target_torque_seq[i]
            rest_ft = self.solve_force(target_force, target_torque, np.array(obj_contacts), qpos[-7:-4])
            rest_ft_arr.append(rest_ft)
        rest_ft_arr = np.array(rest_ft_arr) * 1 / self.obj_mass
        rest_ft_arr[np.where(rest_ft_arr > 0.01)] = 1
        rest_ft_arr[np.where(rest_ft_arr < 0.01)] = 0
        return rest_ft_arr


def cal_IoU(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    _, binary1 = cv2.threshold(image1, 1, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(image2, 1, 255, cv2.THRESH_BINARY)

    binary1 = cv2.bitwise_not(binary1)
    binary2 = cv2.bitwise_not(binary2)

    intersection = cv2.bitwise_and(binary1, binary2)
    union = cv2.bitwise_or(binary1, binary2)

    intersection_pixels = cv2.countNonZero(intersection)
    union_pixels = cv2.countNonZero(union)

    iou = intersection_pixels / union_pixels

    return iou

class PrecisionMetrics:
    def __init__(self, qpos_seq, ndof, hand_model_fn, obj_model_fn):
        self.qpos_seq = qpos_seq
        self.ndof = ndof

        # process hand model fn and object model fn
        dirs = hand_model_fn.split('/')
        dirs[-1] = 'sphere_mesh_display_white_black.xml'
        self.hand_model_fn = '/'.join(dirs)
        dirs = obj_model_fn.split('/')
        dirs[-1] = dirs[-1].replace('light', 'dark')
        self.obj_name = dirs[-1].split('_')[0].capitalize()
        self.obj_model_fn = '/'.join(dirs)

        display_xml, _ = merge_model(self.hand_model_fn, self.obj_model_fn)
        with open('eval_model_tmp.xml', 'w') as f:
            f.write(display_xml)
        self.model = mjpy.load_model_from_path('eval_model_tmp.xml')
        self.sim = mjpy.MjSim(self.model)

        self.color_offset_dict = {
            'Box': 393,
            'Bottle': 684,
            'Banana': 532
        }

        self.tips_map = {
            'index': 0,
            'middle': 1,
            'pinky': 2,
            'ring': 3,
            'thumb': 4
        }

    def obtain_pred_tips(self):
        tip_body_idx = [6, 10, 14, 18, 22]
        # Front Camera
        cam_mat = np.array([
            [1, 0, 0, -0.0257],
            [0, 0, -1, -0.004077],
            [0, 1, 0, 0.7],
            [0, 0, 0, 1]
        ])
        fx = fy = 613.296
        cx = 314.159
        cy = 234.363
        all_tps = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:self.ndof] = qpos[:self.ndof]
            self.sim.forward()
            tip_kps = self.sim.data.body_xpos[tip_body_idx]
            tip_kps = np.matmul(tip_kps - cam_mat[:3, 3], cam_mat[:3, :3])
            u = -fx * tip_kps[:, 0] / tip_kps[:, 2] + cx
            v = fy * tip_kps[:, 1] / tip_kps[:, 2] + cy
            all_tps.append(np.stack([u, v]).T)

        return np.stack(all_tps)

    def eval_tips_error(self):
        label_root_dir = 'sample_data/SingleDepth/Label/Keypoints/{}'.format(self.obj_name)
        label_file_list = glob.glob(label_root_dir + '/front/*.json')
        pred_tips = self.obtain_pred_tips()

        start_idx = 0
        end_idx = len(pred_tips)
        color_offset = self.color_offset_dict[self.obj_name]

        all_pixel_err = []
        label_count = 0

        for i, fn in enumerate(label_file_list):
            frame_id = int(fn[-9:-5])
            frame_id = frame_id - color_offset
            if frame_id <= start_idx or frame_id >= end_idx:
                continue
            labels = json.load(open(fn, 'r'))
            pixel_err = 0
            if len(labels['shapes']) == 0:
                continue
            for s in labels['shapes']:
                name = s['label']
                tip_idx = self.tips_map[name]
                gt = np.array(s['points'])
                ref = pred_tips[frame_id][tip_idx]
                pixel_err += np.linalg.norm(ref - gt)

            pixel_err /= 5
            label_count += 1

            # label_count += len(labels['shapes'])
            all_pixel_err.append(pixel_err)

        all_pixel_err = np.array(all_pixel_err)
        avg_pixel_err = np.sum(all_pixel_err, axis=0) / label_count

        return avg_pixel_err

    def obtain_pred_mask(self):
        viewer = mjpy.MjViewer(self.sim)
        viewer.cam.fixedcamid += 1
        viewer.cam.type = mjpy.const.CAMERA_FIXED
        mask_list = []
        for qpos in self.qpos_seq:
            self.sim.data.qpos[:self.ndof] = qpos[:self.ndof]
            self.sim.data.qpos[2 * self.ndof: 2 * self.ndof + 7] = qpos[self.ndof: self.ndof + 7]
            self.sim.forward()
            img = viewer._read_pixels_as_in_window((640, 480))
            mask_list.append(img[:, :, ::-1])
        return mask_list

    def eval_obj_error(self):
        label_root = 'sample_data/SingleDepth/Label/Mask/{}/front/Masks'.format(self.obj_name)
        label_file_list = glob.glob(label_root + '/*.png')
        pred_mask = self.obtain_pred_mask()
        all_IoU = []
        for fn in label_file_list:
            frame_id = int(fn[-8:-4])
            frame_id = frame_id - self.color_offset_dict[self.obj_name]
            start_idx = 0
            end_idx = len(pred_mask)
            if frame_id <= start_idx or frame_id >= end_idx:
                continue

            label_mask = cv2.imread(fn)
            IoU = cal_IoU(label_mask, pred_mask[frame_id])
            all_IoU.append(IoU)

        all_IoU = np.array(all_IoU)
        avg_IoU = all_IoU.mean()

        return avg_IoU