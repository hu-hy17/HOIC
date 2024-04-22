# -*- coding: utf-8 -*-
import argparse
import cProfile
import os
import sys

import cv2

os.add_dll_directory(os.path.abspath("mujoco210//bin"))

import mujoco_py as mjpy
import matplotlib.pyplot as plt
from mujoco_py.generated import const
import tqdm

from uhc.data_loaders.dataset_grab import DatasetGRAB
from uhc.data_loaders.dataset_grab_new import DatasetGRABNew
from uhc.envs.ho_im4 import HandObjMimic4
from uhc.envs.ho_im4_new import HandObjMimic4New
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
            ele.attrib['group'] = "2"

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

    # display config
    is_loop = True
    is_record = False
    is_eval = False
    is_show_contact = True
    is_viz_force = False
    is_viz_force_score = False
    is_view_gt = False       # Only for dexycb
    is_view_value = False
    camera_view_id = 1
    is_reset = True
    reset_threshold = 12

    # setup display
    if cfg.data_specs['with_obj']:
        display_xml = load_display_xml(cfg)
        with open('display_model_tmp.xml', 'w') as f:
            f.write(display_xml)
        # display_model = mjpy.load_model_from_xml(display_xml)
        display_model = mjpy.load_model_from_path('display_model_tmp.xml')
        display_sim = mjpy.MjSim(display_model)
        display_sim2 = mjpy.MjSim(display_model)
    else:
        display_model = mjpy.load_model_from_path(cfg.vis_model_file)
        display_sim = mjpy.MjSim(display_model)

    # setup env
    data_loader = DatasetGRABNew(cfg.mujoco_model_file, cfg.data_specs, noise=0, mode='test')
    expert_seq = data_loader.load_seq(0, full_seq=True)
    env = HandObjMimic4New(cfg, expert_seq, data_loader.model_path, cfg.data_specs, mode="test")

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

    for seq_idx in tqdm.tqdm(range(1, 2, 1)):
        seq_len = data_loader.get_len(seq_idx)

        all_obj_init_pos_seq.append(data_loader.get_obj_init_pos(seq_idx).numpy())

        # expert_seq = data_loader.load_seq(seq_idx, start_idx=550, full_seq=False, end_idx=650)
        expert_seq = data_loader.load_seq(seq_idx, start_idx=0, full_seq=True)
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

                avg_cps_seq.append(np.concatenate([np.stack(x) for x in env.contact_frame_arr if len(x) != 0])[:, :3])
                # avg_cps_seq.append(env.avg_cps)
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
        # Eval vel and acc
        ###############################
        # obj_vel_arr = np.linalg.norm(np.stack(env.motion_data.obj_vel)[:, 0:3], axis=-1)
        # obj_acc_arr = np.linalg.norm(np.stack(env.motion_data.obj_acc)[:, 0:3], axis=-1)
        # obj_ang_vel_arr = np.linalg.norm(np.stack(env.motion_data.obj_vel)[:, 3:6], axis=-1)
        # obj_ang_acc_arr = np.linalg.norm(np.stack(env.motion_data.obj_acc)[:, 3:6], axis=-1)
        #
        # avg_obj_vel_arr = np.linalg.norm(
        #     np.mean(np.stack(env.motion_data.obj_vel)[:, 0:3].reshape(-1, 15, 3), axis=1), axis=-1)
        # avg_obj_acc_arr = np.linalg.norm(
        #     np.mean(np.stack(env.motion_data.obj_acc)[:, 0:3].reshape(-1, 15, 3), axis=1), axis=-1)
        # avg_obj_ang_vel_arr = np.linalg.norm(
        #     np.mean(np.stack(env.motion_data.obj_vel)[:, 3:6].reshape(-1, 15, 3), axis=1), axis=-1)
        # avg_obj_ang_acc_arr = np.linalg.norm(
        #     np.mean(np.stack(env.motion_data.obj_acc)[:, 3:6].reshape(-1, 15, 3), axis=1), axis=-1)
        #
        # e_obj_vel_arr = np.linalg.norm(expert_seq["obj_vel_seq"], axis=-1)
        # e_obj_ang_vel_arr = np.linalg.norm(expert_seq["obj_angle_vel_seq"], axis=-1)
        # e_obj_acc_arr = np.linalg.norm(np.diff(expert_seq["obj_vel_seq"]), axis=-1) * 30
        # e_obj_ang_acc_arr = np.linalg.norm(np.diff(expert_seq["obj_angle_vel_seq"]), axis=-1) * 30
        #
        # plt.figure()
        # plt.subplot(3, 4, 1)
        # plt.title('vel')
        # plt.plot(obj_vel_arr)
        # plt.subplot(3, 4, 2)
        # plt.title('acc')
        # plt.plot(obj_acc_arr)
        # plt.subplot(3, 4, 3)
        # plt.title('ang vel')
        # plt.plot(obj_ang_vel_arr)
        # plt.subplot(3, 4, 4)
        # plt.title('ang acc')
        # plt.plot(obj_ang_acc_arr)
        #
        # plt.subplot(3, 4, 5)
        # plt.title('e vel')
        # plt.plot(e_obj_vel_arr)
        # plt.subplot(3, 4, 6)
        # plt.title('e acc')
        # plt.plot(e_obj_acc_arr)
        # plt.subplot(3, 4, 7)
        # plt.title('e ang vel')
        # plt.plot(e_obj_ang_vel_arr)
        # plt.subplot(3, 4, 8)
        # plt.title('e ang acc')
        # plt.plot(e_obj_ang_acc_arr)
        #
        # plt.subplot(3, 4, 9)
        # plt.title('avg vel')
        # plt.plot(avg_obj_vel_arr)
        # plt.subplot(3, 4, 10)
        # plt.title('avg acc')
        # plt.plot(avg_obj_acc_arr)
        # plt.subplot(3, 4, 11)
        # plt.title('avg ang vel')
        # plt.plot(avg_obj_ang_vel_arr)
        # plt.subplot(3, 4, 12)
        # plt.title('avg ang acc')
        # plt.plot(avg_obj_ang_acc_arr)
        # plt.show()

        ###############################
        # Force Visualize
        ###############################
        if is_viz_force:
            viz = ForceVisualizer(obj_mesh_fn='assets/SingleDepth/Collision/box.stl',
                                  mano_path='data/mano/models',
                                  root_offset=data_loader.get_obj_init_pos(seq_idx).numpy())
            viz.render(h_kps=np.stack(env.motion_data.hand_kps),
                       obj_pose=np.stack(env.motion_data.obj_pose),
                       cps=env.motion_data.contact_frames,
                       cpf=env.motion_data.contact_force,
                       cpb=env.motion_data.contact_body,
                       cps_ft=env.motion_data.compensate_ft,
                       rfc_rwd=rfc_reward)

    ###############################
    # Eval
    ###############################
    if is_eval:
        test_avg_pene_depth = [0, 0]
        test_cp_num = [0, 0]
        test_rf = [0, 0]
        test_rt = [0, 0]
        test_frame_num = 0

        for idx in range(len(all_pred_qpos_seq)):
            pred_qpos_seq = all_pred_qpos_seq[idx]
            sim_step = pred_qpos_seq.shape[0]
            ref_qpos_seq = np.concatenate([all_ref_qpos_seq[idx][:sim_step], all_obj_ref_pose_seq[idx][:sim_step]],
                                          axis=1)
            phy_met_pred = PhysMetrics(env.model, pred_qpos_seq)
            phy_met_ref = PhysMetrics(env.model, ref_qpos_seq)

            pred_pene = np.array(phy_met_pred.eval_penetration())
            ref_pene = np.array(phy_met_ref.eval_penetration())
            pred_cp_num = np.array(phy_met_pred.eval_contact_point())
            ref_cp_num = np.array(phy_met_ref.eval_contact_point())
            pred_rf, pred_rt = phy_met_pred.eval_stable()
            ref_rf, ref_rt = phy_met_ref.eval_stable()

            test_frame_num += sim_step
            test_avg_pene_depth[0] += np.sum(pred_pene)
            test_avg_pene_depth[1] += np.sum(ref_pene)
            test_cp_num[0] += np.sum(pred_cp_num)
            test_cp_num[1] += np.sum(ref_cp_num)
            test_rf[0] += np.sum(pred_rf)
            test_rf[1] += np.sum(ref_rf)
            test_rt[0] += np.sum(pred_rt)
            test_rt[1] += np.sum(ref_rt)

            print('Seq Idx: ' + str(idx))
            print('Avg Pene Depth(Pred/Ref): ' + str(np.mean(pred_pene)) + '/' + str(np.mean(ref_pene)))
            print('Avg Cp Num(Pred/Ref): ' + str(np.mean(pred_cp_num)) + '/' + str(np.mean(ref_cp_num)))
            print('Avg Rest Force(Pred/Ref)' + str(np.mean(pred_rf)) + '/' + str(np.mean(ref_rf)))
            print('Avg Rest Torque(Pred/Ref)' + str(np.mean(pred_rt)) + '/' + str(np.mean(ref_rt)))

        print('Total Avg Pene Depth(Pred/Ref): ' + str(test_avg_pene_depth[0] / test_frame_num) + '/' +
              str(test_avg_pene_depth[1] / test_frame_num))
        print('Total Avg Cp Num(Pred/Ref): ' + str(test_cp_num[0] / test_frame_num) + '/' +
              str(test_cp_num[1] / test_frame_num))
        print('Total Avg Rest Force(Pred/Ref)' + str(test_rf[0] / test_frame_num) + '/' +
              str(test_rf[1] / test_frame_num))
        print('Total Avg Rest Torque(Pred/Ref)' + str(test_rt[0] / test_frame_num) + '/' +
              str(test_rt[1] / test_frame_num))

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
                viewer.add_marker(pos=np.array([0, 0.2, 0.7]),
                                  label=str(idx),
                                  rgba=np.array([1.0, 0, 0, 1.0]),
                                  size=[0.01, 0.01, 0.01])
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
