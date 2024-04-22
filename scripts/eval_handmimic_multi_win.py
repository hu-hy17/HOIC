# -*- coding: utf-8 -*-
import argparse
import cProfile
import os
import sys
import time

import cv2

import platform
if platform.system() == 'Windows':
    os.add_dll_directory(os.path.abspath("mujoco210//bin"))

import mujoco_py as mjpy
import matplotlib.pyplot as plt
from mujoco_py.generated import const
import tqdm

from uhc.data_loaders.dataset_singledepth import DatasetSingleDepth
from uhc.envs.ho_im4 import HandObjMimic4, MotionData
from uhc.envs.ho_reward import *
from uhc.khrylib.rl.core import PolicyGaussian
from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.core.critic import Value
from uhc.utils.tools import CustomUnpickler
from uhc.utils.torch_utils import *
from visualize import ForceVisualizer

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np

from uhc.utils.config_utils.handmimic_config import Config
from uhc.data_loaders.mjxml.MujocoXML import MujocoXML
from metrics import PhysMetrics


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

    # get object mesh file
    obj_mesh_fn = None
    for c in obj_model.asset.getchildren():
        if c.get('name').startswith('V_'):
            obj_mesh_fn = c.get('file')
            break
    if obj_mesh_fn is None:
        print("Error: No Object Mesh file Found!")
        raise FileNotFoundError

    return hand_model.get_xml(), obj_mesh_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    # display config
    is_loop = False
    is_record = True
    is_eval = False
    is_show_contact = False
    is_viz_force = False
    is_viz_force_score = False
    is_view_gt = False  # Only for dexycb
    is_view_value = False
    camera_view_id = 1
    is_reset = False
    reset_threshold = 12
    dataset_mode = 'train'

    # setup display
    display_xml, obj_mesh_fn = load_display_xml(cfg)
    with open('display_model_tmp.xml', 'w') as f:
        f.write(display_xml)
    # display_model = mjpy.load_model_from_xml(display_xml)
    display_model = mjpy.load_model_from_path('display_model_tmp.xml')
    display_sim = mjpy.MjSim(display_model)
    display_sim2 = mjpy.MjSim(display_model)

    # setup env
    data_loader = DatasetGRAB(cfg.mujoco_model_file, cfg.data_specs, noise=0, mode=dataset_mode)
    expert_seq = data_loader.load_seq(0, full_seq=True)
    # env = HandObjMimic3(cfg, expert_seq, data_loader.model_path, cfg.data_specs, mode="test")
    env = HandObjMimic4(cfg, expert_seq, data_loader.model_path, cfg.data_specs, mode="test")
    # env = HandObjMimic2(cfg, expert_seq, data_loader.model_path, cfg.data_specs, mode="test")

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

    policy_net_params_arr = []
    for i in range(3):
        policy_net_params_arr.append(policy_net.net.affine_layers[i].weight.data.numpy())

    running_state = model_cp["running_state"]

    all_pred_qpos_seq = []
    all_ref_qpos_seq = []
    all_obj_ref_pose_seq = []
    all_gt_qpos_seq = []
    all_gt_obj_pose_seq = []
    all_obj_init_pos_seq = []
    all_target_hand_pose_seq = []
    all_vf_reward_seq = []
    all_avg_cps_seq = []

    reward_func = reward_list[cfg.reward_type - 1]
    seq_step = 1 if dataset_mode == 'test' else 4
    # seq_step = 1

    for seq_idx in tqdm.tqdm(range(1, 2, seq_step)):
        seq_len = data_loader.get_len(seq_idx)

        all_obj_init_pos_seq.append(data_loader.get_obj_init_pos(seq_idx).numpy())

        # expert_seq = data_loader.load_seq(seq_idx, start_idx=550, full_seq=False, end_idx=650)
        expert_seq = data_loader.load_seq(seq_idx, full_seq=True)
        env.set_expert(expert_seq)
        pred_qpos_seq = []
        pred_joint_pos = []
        tot_reward_seq = []
        all_reward_seq = []
        avg_cps_seq = []

        value_seq = []
        target_pose_seq = []
        pred_qpos_seq.append(env.data.qpos.copy())
        ref_qpos_seq = expert_seq["hand_dof_seq"].copy()
        ref_obj_pose_seq = expert_seq["obj_pose_seq"].copy()
        if is_view_gt:
            gt_qpos_seq = expert_seq["gt_hand_dof_seq"].copy()
            gt_obj_pose_seq = expert_seq["gt_obj_pose_seq"].copy()
        ref_joint_pos = expert_seq["body_pos_seq"].copy()
        gt_joint_pos = expert_seq["raw_body_pos_seq"].copy()
        obs = env.reset()
        pred_joint_pos.append(env.get_wbody_pos().copy().reshape(-1, 3))

        with torch.no_grad():
            while True:
                if running_state is not None:
                    obs = running_state(obs, update=False)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = policy_net.select_action(obs_tensor, mean_action=True)[0].cpu().numpy()
                value = value_net(obs_tensor)
                action = action.astype(np.float64)
                obs, reward, done, info = env.step(action)
                if value < reset_threshold and is_reset:
                    obs = env.reset(True)

                # avg_cps_seq.append(np.concatenate([np.stack(x) for x in env.contact_frame_arr if len(x) != 0])[:, :3])
                avg_cps_seq.append(env.avg_cps)
                target_pose_seq.append(env.target_hand_pose)
                tot_rew, all_rew = reward_func(env, obs, action, info)
                tot_reward_seq.append(tot_rew)
                all_reward_seq.append(all_rew)
                value_seq.append(value)

                pred_qpos_seq.append(env.data.qpos.copy())
                pred_joint_pos.append(env.get_wbody_pos().copy().reshape(-1, 3))

                if done:
                    break
        target_pose_seq = np.stack(target_pose_seq)

        # Compute average reward
        tot_reward_seq = np.stack(tot_reward_seq)
        all_reward_seq = np.stack(all_reward_seq)

        total_step = ref_joint_pos.shape[0]

        print("Avg tot reward: %.4f" % np.mean(tot_reward_seq[:-1]))
        print("Avg all reward: ", np.mean(all_reward_seq, axis=0))

        rfc_reward = all_reward_seq[:, -2]

        # # Compute MPJPE, object pos error and orientation error
        pred_qpos_seq = np.stack(pred_qpos_seq)
        pred_joint_pos = np.stack(pred_joint_pos)
        #
        sim_step = pred_joint_pos.shape[0]

        print(str(sim_step) + '/' + str(total_step - 5))
        gt_joint_pos = gt_joint_pos[:sim_step]
        ref_joint_pos = ref_joint_pos[:sim_step]
        ref_obj_pose_seq = ref_obj_pose_seq[:sim_step]
        #
        mpjpe = np.mean(np.linalg.norm(pred_joint_pos - gt_joint_pos, axis=2), axis=0)[0]
        print("MPJPE(Pred Vs GT): %.4f" % mpjpe)
        mpjpe = np.mean(np.linalg.norm(ref_joint_pos - gt_joint_pos, axis=2))
        print("MPJPE(Ref Vs GT): %.4f" % mpjpe)
        obj_pos_err = pred_qpos_seq[:, env.ndof:env.ndof + 3] - ref_obj_pose_seq[:, :3]
        obj_pos_err = np.linalg.norm(obj_pos_err, axis=1)
        print("Obj pos err: %.4f" % np.mean(obj_pos_err))
        pred_obj_rot = torch.Tensor(pred_qpos_seq[:, env.ndof + 3:])
        gt_obj_rot = torch.Tensor(ref_obj_pose_seq[:, 3:])
        obj_quat_diff = quaternion_multiply_batch(gt_obj_rot, quaternion_inverse_batch(pred_obj_rot))
        obj_rot_diff = 2.0 * torch.arcsin(torch.clip(torch.norm(obj_quat_diff[:, 1:], dim=-1), 0, 1))
        obj_rot_diff = obj_rot_diff.cpu().numpy()
        print("Obj rot err: %.4f" % np.mean(obj_rot_diff))

        # plot value
        if is_view_value:
            plt.plot(value_seq)
            plt.show()

        # ref_qpos_seq[:, 0] += 0.2
        # ref_obj_pose_seq[:, 0] += 0.2

        all_pred_qpos_seq.append(pred_qpos_seq)
        all_ref_qpos_seq.append(ref_qpos_seq)
        all_obj_ref_pose_seq.append(ref_obj_pose_seq)
        all_target_hand_pose_seq.append(target_pose_seq)
        all_vf_reward_seq.append(rfc_reward)
        all_avg_cps_seq.append(avg_cps_seq)
        if is_view_gt:
            all_gt_qpos_seq.append(gt_qpos_seq)
            all_gt_obj_pose_seq.append(gt_obj_pose_seq)

        ###############################
        # Force Visualize
        ###############################
        if is_viz_force:
            viz = ForceVisualizer(obj_mesh_fn=obj_mesh_fn,
                                  mano_path='data/mano/models',
                                  root_offset=data_loader.get_obj_init_pos(seq_idx).numpy(),
                                  add_marker=False)
            # viz.render(h_kps=np.stack(env.motion_data.hand_kps),
            #            obj_pose=np.stack(env.motion_data.obj_pose),
            #            cps=env.motion_data.contact_frames,
            #            cpf=env.motion_data.contact_force,
            #            cpb=env.motion_data.contact_body,
            #            cps_ft=env.motion_data.compensate_ft,
            #            rfc_rwd=rfc_reward)

            render_save_dir = 'results/img/render/mimic/' + str(seq_idx)
            if not os.path.exists(render_save_dir):
                os.makedirs(render_save_dir)
            frame_num = len(env.motion_data.hand_kps)
            viz.render_skeleton(h_kps=np.stack(env.motion_data.hand_kps)[np.arange(0, frame_num, 15)],
                                obj_pose=np.stack(env.motion_data.obj_pose)[np.arange(0, frame_num, 15)],
                                save=True, save_dir=render_save_dir)

            render_save_dir = 'results/img/render/ref/' + str(seq_idx)
            if not os.path.exists(render_save_dir):
                os.makedirs(render_save_dir)
            viz.render_skeleton(h_kps=expert_seq.get('body_pos_seq'),
                                obj_pose=expert_seq.get('obj_pose_seq'),
                                save=True, save_dir=render_save_dir)

    ###############################
    # Eval
    ###############################
    if is_eval:
        test_avg_pene_depth = [0, 0]
        test_cp_num = [0, 0]
        test_rft = [0, 0]
        test_hand_acc = [0, 0]
        test_obj_acc = [0, 0]
        test_obj_ang_acc = [0, 0]
        test_frame_num = 0

        for idx in range(len(all_pred_qpos_seq)):
            pred_qpos_seq = all_pred_qpos_seq[idx]
            sim_step = pred_qpos_seq.shape[0]
            ref_qpos_seq = np.concatenate([all_ref_qpos_seq[idx][:sim_step], all_obj_ref_pose_seq[idx][:sim_step]],
                                          axis=1)
            phy_met_pred = PhysMetrics(env.model, pred_qpos_seq, obj_mesh_fn)
            phy_met_ref = PhysMetrics(env.model, ref_qpos_seq, obj_mesh_fn)

            pred_hand_acc, pred_obj_acc, pred_obj_ang_acc = phy_met_pred.eval_jitter()
            ref_hand_acc, ref_obj_acc, ref_obj_ang_acc = phy_met_ref.eval_jitter()
            pred_pene = np.array(phy_met_pred.eval_penetration())
            ref_pene = np.array(phy_met_ref.eval_penetration())
            pred_cp_num = np.array(phy_met_pred.eval_contact_point())
            ref_cp_num = np.array(phy_met_ref.eval_contact_point())
            pred_rft = phy_met_pred.eval_stable()
            ref_rft = phy_met_ref.eval_stable()

            # np.save('sa22_phys.npy', ref_rft)

            test_frame_num += sim_step
            test_avg_pene_depth[0] += np.sum(pred_pene)
            test_avg_pene_depth[1] += np.sum(ref_pene)
            test_cp_num[0] += np.sum(pred_cp_num)
            test_cp_num[1] += np.sum(ref_cp_num)
            test_rft[0] += np.sum(pred_rft)
            test_rft[1] += np.sum(ref_rft)

            test_hand_acc[0] += sim_step * pred_hand_acc
            test_hand_acc[1] += sim_step * ref_hand_acc
            test_obj_acc[0] += sim_step * pred_obj_acc
            test_obj_acc[1] += sim_step * ref_obj_acc
            test_obj_ang_acc[0] += sim_step * pred_obj_ang_acc
            test_obj_ang_acc[1] += sim_step * ref_obj_ang_acc

            print('Seq Idx: ' + str(idx))
            print('Avg Pene Depth(Pred/Ref): ' + str(np.mean(pred_pene)) + '/' + str(np.mean(ref_pene)))
            print('Avg Cp Num(Pred/Ref): ' + str(np.mean(pred_cp_num)) + '/' + str(np.mean(ref_cp_num)))
            print('Avg Phys Plausible Frame Ratio(Pred/Ref)' + str(1 - np.mean(pred_rft)) + '/' + str(
                1 - np.mean(ref_rft)))
            print('Avg hand acc(Pred/Ref)' + str(pred_hand_acc) + '/' + str(ref_hand_acc))
            print('Avg obj acc(Pred/Ref)' + str(pred_obj_acc) + '/' + str(ref_obj_acc))
            print('Avg obj ang acc(Pred/Ref)' + str(pred_obj_ang_acc) + '/' + str(ref_obj_ang_acc))

        print('Total Avg Pene Depth(Pred/Ref): ' + str(test_avg_pene_depth[0] / test_frame_num) + '/' +
              str(test_avg_pene_depth[1] / test_frame_num))
        print('Total Avg Cp Num(Pred/Ref): ' + str(test_cp_num[0] / test_frame_num) + '/' +
              str(test_cp_num[1] / test_frame_num))
        print('Total Phys Plausible Frame Ratio(Pred/Ref)' + str(1 - test_rft[0] / test_frame_num) + '/' +
              str(1 - test_rft[1] / test_frame_num))
        print('Total hand acc(Pred/Ref): ' + str(test_hand_acc[0] / test_frame_num) + '/' +
              str(test_hand_acc[1] / test_frame_num))
        print('Total obj acc(Pred/Ref): ' + str(test_obj_acc[0] / test_frame_num) + '/' +
              str(test_obj_acc[1] / test_frame_num))
        print('Total obj ang acc(Pred/Ref): ' + str(test_obj_ang_acc[0] / test_frame_num) + '/' +
              str(test_obj_ang_acc[1] / test_frame_num))

    ###############################
    # Display
    ###############################
    viewer = mjpy.MjViewer(display_sim)
    viewer2 = mjpy.MjViewer(display_sim2)


    # setup camera
    def setup_cam(viewer):
        # viewer.cam.trackbodyid = 3
        # viewer.cam.distance = 0.4
        # viewer.cam.lookat[0] = 0.05
        # viewer.cam.lookat[1] = 0
        # viewer.cam.lookat[2] = 0.7
        viewer.cam.fixedcamid += camera_view_id
        viewer.cam.type = const.CAMERA_FIXED


    setup_cam(viewer)
    setup_cam(viewer2)

    video_size = (640, 480)

    loop_count = 100000 if is_loop else len(all_pred_qpos_seq)
    for seq_idx in range(loop_count):
        seq_idx = seq_idx % len(all_pred_qpos_seq)
        root_offset = all_obj_init_pos_seq[seq_idx].copy()
        pred_qpos_seq = all_pred_qpos_seq[seq_idx].copy()
        ref_qpos_seq = all_ref_qpos_seq[seq_idx].copy()
        ref_obj_pose_seq = all_obj_ref_pose_seq[seq_idx].copy()
        target_hand_pose_seq = all_target_hand_pose_seq[seq_idx].copy()

        if is_view_gt:
            gt_qpos_seq = all_gt_qpos_seq[seq_idx].copy()
            gt_obj_pose_seq = all_gt_obj_pose_seq[seq_idx].copy()

        pred_qpos_seq[:, :3] += root_offset
        target_hand_pose_seq[:, :3] += root_offset
        pred_qpos_seq[:, env.hand_qpos_dim: env.hand_qpos_dim + 3] += root_offset
        ref_qpos_seq[:, :3] += root_offset
        ref_obj_pose_seq[:, :3] += root_offset

        # output 2D key points position
        output_folder = 'results/keypoints'
        tip_body_idx = [6, 10, 14, 18, 22]
        # Front Camera
        cam_mat = np.array([
            [1, 0, 0, -0.0257],
            [0, 0, -1, -0.004077],
            [0, 1, 0, 0.7],
            [0, 0, 0, 1]
        ])
        # Back Camera
        # cam_mat = np.array([
        #     [-0.941048, -0.10347, -0.322058, -0.10419777],
        #     [-0.323253, -0.005463, 0.946296, 0.84889326],
        #     [-0.099672, 0.994618, -0.028306, 0.75024971],
        #     [0, 0, 0, 1]
        # ])
        fx = fy = 613.296
        cx = 314.159
        cy = 234.363
        all_pred_tps = []
        all_ref_tps = []
        for qpos in pred_qpos_seq:
            display_sim.data.qpos[:env.hand_qpos_dim] = qpos[:env.hand_qpos_dim]
            display_sim.forward()
            tip_kps = display_sim.data.body_xpos[tip_body_idx]
            tip_kps = np.matmul(tip_kps - cam_mat[:3, 3], cam_mat[:3, :3])
            u = -fx * tip_kps[:, 0] / tip_kps[:, 2] + cx
            v = fy * tip_kps[:, 1] / tip_kps[:, 2] + cy
            # canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
            # for k in range(5):
            #     x, y = round(u[k]), round(v[k])
            #     cv2.circle(canvas, (x, y), 3, (0, 0, 0), -1)
            # cv2.imshow("Canvas", canvas)
            # cv2.waitKey(30)
            all_pred_tps.append(np.stack([u, v]).T)
        for qpos in ref_qpos_seq:
            display_sim.data.qpos[:env.hand_qpos_dim] = qpos[:env.hand_qpos_dim]
            display_sim.forward()
            tip_kps = display_sim.data.body_xpos[tip_body_idx]
            tip_kps = np.matmul(tip_kps - cam_mat[:3, 3], cam_mat[:3, :3])
            u = -fx * tip_kps[:, 0] / tip_kps[:, 2] + cx
            v = fy * tip_kps[:, 1] / tip_kps[:, 2] + cy
            # canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
            # for k in range(5):
            #     x, y = round(u[k]), round(v[k])
            #     cv2.circle(canvas, (x, y), 3, (0, 0, 0), -1)
            # cv2.imshow("Canvas", canvas)
            # cv2.waitKey(30)
            all_ref_tps.append(np.stack([u, v]).T)
        np.save(os.path.join(output_folder, 'pred_tps.npy'), np.stack(all_pred_tps))
        np.save(os.path.join(output_folder, 'ref_tps.npy'), np.stack(all_ref_tps))

        # create output folder
        output_folder = 'results/img/' + str(seq_idx)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for idx in range(len(pred_qpos_seq) - 1):
            cur_t = idx
            display_sim.data.qpos[:env.hand_qpos_dim] = pred_qpos_seq[cur_t][:env.hand_qpos_dim]
            if is_view_gt:
                display_sim2.data.qpos[env.hand_qpos_dim: 2 * env.hand_qpos_dim] = gt_qpos_seq[cur_t]
            else:
                display_sim2.data.qpos[env.hand_qpos_dim: 2 * env.hand_qpos_dim] = ref_qpos_seq[cur_t]
            # display_sim2.data.qpos[env.hand_qpos_dim: 2 * env.hand_qpos_dim] = target_hand_pose_seq[cur_t]

            display_sim.data.qpos[2 * env.hand_qpos_dim: 2 * env.hand_qpos_dim + 7] = pred_qpos_seq[cur_t][
                                                                                      env.hand_qpos_dim:]
            if is_view_gt:
                display_sim2.data.qpos[2 * env.hand_qpos_dim + 7:] = gt_obj_pose_seq[cur_t]
            else:
                display_sim2.data.qpos[2 * env.hand_qpos_dim + 7:] = ref_obj_pose_seq[cur_t]
            # display_sim2.data.qpos[2 * env.hand_qpos_dim + 7:] = pred_qpos_seq[cur_t][env.hand_qpos_dim:]

            display_sim.forward()
            display_sim2.forward()

            # Add markers
            ref_contact_arr = []
            contact_arr = []
            if is_show_contact:
                for contact in display_sim2.data.contact[:display_sim2.data.ncon]:
                    g1, g2 = contact.geom1, contact.geom2
                    if ref_hand_geom_range[0] <= g1 <= ref_hand_geom_range[1] and \
                            ref_obj_geom_range[0] <= g2 <= ref_obj_geom_range[1]:
                        ref_contact_arr.append(contact.pos.copy())
                contact_arr = all_avg_cps_seq[seq_idx][idx]

            vf_score = all_vf_reward_seq[seq_idx][idx - 1] if idx > 0 else 1

            if not is_record:
                viewer._markers[:] = []
                viewer2._markers[:] = []
                viewer.add_marker(pos=np.array([0, 0.2, 0.7]),
                                  label=str(idx),
                                  rgba=np.array([1.0, 0, 0, 1.0]),
                                  size=[0.001, 0.001, 0.001])
                viewer.add_marker(pos=np.array([0.12, 0.3, 0.7 + 2 / 20]),
                                  label='',
                                  rgba=np.array([1.0, 0, 0.0, 1.0]),
                                  size=[0.01, 0.01, 0.001])
                viewer.add_marker(pos=np.array([0.12, 0.3, 0.7 + vf_score / 20]),
                                  label='',
                                  rgba=np.array([0.0, 0, 1.0, 1.0]),
                                  size=[0.01, 0.01, vf_score / 20])
                if is_show_contact:
                    for c in contact_arr:
                        viewer.add_marker(pos=c[:3] + root_offset,
                                          label='',
                                          rgba=np.array([1.0, 0, 0, 1.0]),
                                          size=[0.001, 0.001, 0.001])
                    for c in ref_contact_arr:
                        viewer2.add_marker(pos=c,
                                           label='',
                                           rgba=np.array([0.0, 0, 1.0, 1.0]),
                                           size=[0.001, 0.001, 0.001])
                viewer.render()
                viewer2.render()
            else:
                # Render ref and mimic
                viewer._markers[:] = []
                # viewer.add_marker(pos=np.array([0, 0.2, 0.7]),
                #                   label=str(idx),
                #                   rgba=np.array([1.0, 0, 0, 1.0]),
                #                   size=[0.001, 0.001, 0.001])
                if is_viz_force_score:
                    viewer.add_marker(pos=np.array([0.12, 0.3, 0.7 + 2 / 20]),
                                      label='',
                                      rgba=np.array([1.0, 0, 0.0, 1.0]),
                                      size=[0.01, 0.01, 0.001])
                    viewer.add_marker(pos=np.array([0.12, 0.3, 0.7 + vf_score / 20]),
                                      label='',
                                      rgba=np.array([0.0, 0, 1.0, 1.0]),
                                      size=[0.01, 0.01, vf_score / 20])
                if is_show_contact:
                    for c in contact_arr:
                        viewer.add_marker(pos=c[:3] + root_offset,
                                          label='',
                                          rgba=np.array([0, 0, 1.0, 1.0]),
                                          size=[0.001, 0.001, 0.001])
                img_mimic = viewer._read_pixels_as_in_window(video_size)
                cv2.imwrite('results/img/' + str(seq_idx) + '/ref_%04d.png' % idx, img_mimic[:, :, ::-1])

                viewer2._markers[:] = []

                if is_show_contact:
                    for c in ref_contact_arr:
                        viewer2.add_marker(pos=c,
                                           label='',
                                           rgba=np.array([1.0, 0, 0, 1.0]),
                                           size=[0.001, 0.001, 0.001])
                img_ref = viewer2._read_pixels_as_in_window(video_size)
                cv2.imwrite('results/img/' + str(seq_idx) + '/mimic_%04d.png' % idx, img_ref[:, :, ::-1])
