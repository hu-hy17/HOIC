import numpy as np
import math
import transforms3d as t3d

from uhc.utils.math_utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler

conf_mix_coef = 0.5
conf_bonus = 0.5

conf_group = [
    [2, 1, 0],  # index (base->top)
    [6, 5, 4],  # middle
    [10, 9, 8],  # little
    [14, 13, 12],  # ring
    [18, 17, 16]  # thumb
]

dof_conf_map = [
    2, 2, 1, 0,  # Index
    6, 6, 5, 4,  # Middle
    10, 10, 9, 8,  # Little
    14, 14, 13, 12,  # Ring
    18, 18, 17, 16  # Thumb
]

dof_finger_idx_map = [
    0, 0, 0, 0,  # Index
    1, 1, 1, 1,  # Middle
    2, 2, 2, 2,  # Little
    3, 3, 3, 3,  # Ring
    4, 4, 4, 4  # Thumb
]

body_conf_map = [
    2, 1, 0, 0,  # Index
    6, 5, 4, 4,  # Middle
    10, 9, 8, 8,  # Little
    14, 13, 12, 12,  # Ring
    18, 17, 16, 16  # Thumb
]

body_finger_idx_map = [
    0, 0, 0, 0,  # Index
    1, 1, 1, 1,  # Middle
    2, 2, 2, 2,  # Little
    3, 3, 3, 3,  # Ring
    4, 4, 4, 4  # Thumb
]


def ho_mimic_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.5)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 100)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = (pose_diff ** 2).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = (wpose_diff ** 2).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity rewards
    vel_dist = cur_qvel - e_qvel
    vel_dist = (vel_dist ** 2).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.power(np.linalg.norm(obj_diff), 2)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = obj_rot_diff[0] ** 2
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = (obj_vel_diff ** 2).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    if info["end"]:
        reward += 1.0 / (1 - env.cc_cfg.gamma) - 1
    # np.set_printoptions(precision=5, suppress=1)
    # print(np.array([reward, vel_dist, pose_diff, wpose_diff]), \
    # np.array([pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]))
    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward]
    )


# Change weight
def ho_mimic_reward_2(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


# Strengthen root pos
def ho_mimic_reward_3(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff)
    wpose_reward = math.exp(-k_wp * wpose_diff.mean())

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1)
    jpos_reward = math.exp(-k_j * jpos_dist.mean())

    # additional reward on hand root
    hand_root_pos_reward = math.exp(-k_j * 10 * jpos_dist[0])
    hand_root_rot_reward = math.exp(-k_wp * 10 * wpose_diff[0])

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward + \
                  0.25 * hand_root_rot_reward + 0.25 * hand_root_pos_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v + 0.5)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward]
    )


# Add root vel regularization
def ho_mimic_reward_4(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff)
    wpose_reward = math.exp(-k_wp * wpose_diff.mean())

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1)
    jpos_reward = math.exp(-k_j * jpos_dist.mean())

    # additional reward on hand root
    root_pos_vel_reg = max(0, np.linalg.norm(cur_qvel[:3]) - 0.03)
    root_ang_vel_reg = max(0, np.linalg.norm(cur_qvel[3:6]) - 0.74)
    hand_root_pos_reward = math.exp(-2 * root_pos_vel_reg)
    hand_root_rot_reward = math.exp(-0.1 * root_ang_vel_reg)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward + \
                  0.1 * hand_root_rot_reward + 0.1 * hand_root_pos_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v + 0.2)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward]
    )


# Add root acc regularization
def ho_mimic_reward_5(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff)
    wpose_reward = math.exp(-k_wp * wpose_diff.mean())

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1)
    jpos_reward = math.exp(-k_j * jpos_dist.mean())

    # additional reward on hand root
    cur_acc = env.data.qacc.copy()
    root_pos_acc_reg = max(0, np.linalg.norm(cur_acc[:3]) - 1.0)
    root_ang_acc_reg = max(0, np.linalg.norm(cur_acc[3:6]) - 20.0)
    hand_root_pos_reward = math.exp(-0.1 * root_pos_acc_reg)
    hand_root_rot_reward = math.exp(-0.05 * root_ang_acc_reg)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward + \
                  0.2 * hand_root_rot_reward + 0.2 * hand_root_pos_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v + 0.4)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


# Add contact reward
def ho_mimic_reward_6(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)
    e_contact_info = env.get_expert_attr("contact_info_seq", ind)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff)
    wpose_reward = math.exp(-k_wp * wpose_diff.mean())

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1)
    jpos_reward = math.exp(-k_j * jpos_dist.mean())

    # additional reward on hand root
    cur_acc = env.data.qacc.copy()
    root_pos_acc_reg = max(0, np.linalg.norm(cur_acc[:3]) - 1.0)
    root_ang_acc_reg = max(0, np.linalg.norm(cur_acc[3:6]) - 20.0)
    hand_root_pos_reward = math.exp(-0.1 * root_pos_acc_reg)
    hand_root_rot_reward = math.exp(-0.005 * root_ang_acc_reg)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = math.exp(-k_orfc * obj_rfc)
    else:
        w_orfc = 0

    # contact reward
    contact_info = env.get_contact_rwd()
    contact_reward = 0
    if len(e_contact_info) == 0:
        contact_reward = 1
    else:
        for key, value in e_contact_info.items():
            contact_pos = contact_info.get(key)
            if contact_pos is not None:
                contact_reward += math.exp(-k_j * np.linalg.norm(contact_pos - value))
        contact_reward /= len(e_contact_info)

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward + \
                  0.1 * hand_root_rot_reward + 0.1 * hand_root_pos_reward + 0.4 * contact_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v + 0.6)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, contact_reward]
    )


# Add root acc regularization
# Another rfc reward
def ho_mimic_reward_7(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff)
    wpose_reward = math.exp(-k_wp * wpose_diff.mean())

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1)
    jpos_reward = math.exp(-k_j * jpos_dist.mean())

    # additional reward on hand root
    cur_acc = env.data.qacc.copy()
    root_pos_acc_reg = max(0, np.linalg.norm(cur_acc[:3]) - 1.0)
    root_ang_acc_reg = max(0, np.linalg.norm(cur_acc[3:6]) - 20.0)
    hand_root_pos_reward = math.exp(-0.1 * root_pos_acc_reg)
    hand_root_rot_reward = math.exp(-0.05 * root_ang_acc_reg)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 1
    if env.cc_cfg.residual_force:
        obj_rfc = env.solve_rfc_torque(action)
        obj_rfc = np.linalg.norm(obj_rfc) ** 2
        obj_rfc_reward = max(0, 1 - k_orfc * (obj_rfc ** 2))
    else:
        w_orfc = 0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward + \
                  0.1 * hand_root_rot_reward + 0.1 * hand_root_pos_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward
    hand_reward /= (w_p + w_wp + w_j + w_v + 0.2)
    obj_reward /= (w_op + w_or + w_ov)

    # overall reward
    reward = hand_reward * obj_reward * obj_rfc_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


def ho_mimic_reward_8(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        obj_rfc_reward = math.exp(-k_orfc * env.rfc_score)
    else:
        obj_rfc_reward = 1.0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov)

    # overall reward
    reward = hand_reward * obj_reward * obj_rfc_reward

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


def ho_mimic_reward_9(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        # vf_norm = np.linalg.norm(action[env.ndof: env.vf_dim]) / math.sqrt(3)
        obj_rfc_reward = math.exp(-k_orfc * env.rfc_score)
    else:
        obj_rfc_reward = 1.0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    # if info["end"]:
    #     reward += 1.0 / (1 - env.cc_cfg.gamma) - 1

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


# With Confidence
def ho_mimic_reward_10(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)
    e_conf = env.get_expert_attr("conf_seq", ind)

    # process conf
    finger_tot_conf = np.array([np.sum(e_conf[ids]) for ids in conf_group])
    wc_dof = (1 - conf_mix_coef) * e_conf[dof_conf_map] + conf_mix_coef * finger_tot_conf[dof_finger_idx_map]
    wc_dof += conf_bonus
    wc_dof = wc_dof / np.mean(wc_dof)
    wc_body = (1 - conf_mix_coef) * e_conf[body_conf_map] + conf_mix_coef * finger_tot_conf[body_finger_idx_map]
    wc_body += conf_bonus
    wc_body = wc_body / np.mean(wc_body)

    wc_dof = np.concatenate([np.ones(6), wc_dof])
    wc_body = np.concatenate([np.ones(1), wc_body])

    # pose reward
    pose_diff = cur_qpos - e_qpos
    pose_dist = np.abs(pose_diff)
    pose_dist_conf = np.mean(wc_dof * pose_dist)
    pose_reward = math.exp(-k_p * pose_dist_conf)

    # body joint reward
    jpos_diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(jpos_diff, axis=1)
    jpos_dist_conf = np.mean(wc_body * jpos_dist)
    jpos_reward = math.exp(-k_j * jpos_dist_conf)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_dist = np.abs(wpose_diff)
    wpose_dist_conf = np.mean(wc_body * wpose_dist)
    wpose_reward = math.exp(-k_wp * wpose_dist_conf)

    # velocity reward
    vel_diff = cur_qvel - e_qvel
    vel_dist = np.abs(vel_diff)
    vel_dist_conf = np.mean(wc_dof * vel_dist)
    vel_reward = math.exp(-k_v * vel_dist_conf)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # object rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        # vf_norm = np.linalg.norm(action[env.ndof: env.vf_dim]) / math.sqrt(3)
        obj_rfc_reward = math.exp(-k_orfc * env.rfc_score)
    else:
        obj_rfc_reward = 1.0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    if info["end"]:
        reward += 1.0 / (1 - env.cc_cfg.gamma) - 1

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


# Remove object z-axis rotation diff
def ho_mimic_reward_11(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_qvel - e_qvel
    vel_dist = np.abs(vel_dist).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    orot1 = t3d.quaternions.quat2mat(obj_pose[3:])
    orot2 = t3d.quaternions.quat2mat(epose_obj[3:])
    z_diff = np.arccos(np.dot(orot1[:, 2], orot2[:, 2]))
    obj_rot_dist = np.abs(z_diff) / 2

    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        # vf_norm = np.linalg.norm(action[env.ndof: env.vf_dim]) / math.sqrt(3)
        obj_rfc_reward = math.exp(-k_orfc * env.rfc_score)
    else:
        obj_rfc_reward = 1.0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    # if info["end"]:
    #     reward += 1.0 / (1 - env.cc_cfg.gamma) - 1

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


def ho_mimic_reward_12(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_op, w_or, w_ov, w_orfc = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_op", 0.45),
        ws.get("w_or", 0.45),
        ws.get("w_ov", 0.1),
        ws.get("w_orfc", 0.2)
    )

    k_p, k_wp, k_v, k_j, k_op, k_or, k_ov, k_orfc = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_op", 100.0),
        ws.get("k_or", 5.0),
        ws.get("k_ov", 0.05),
        ws.get("k_orfc", 1)
    )

    # data from env
    t = env.cur_t + env.start_ind
    ind = t

    # Current policy states
    cur_qpos = env.get_hand_qpos()
    cur_qvel = env.get_hand_qvel()
    cur_wbquat = env.get_wbody_quat().reshape(-1, 4)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("hand_dof_seq", ind)
    e_qvel = env.get_expert_attr("hand_dof_vel_seq", ind)
    e_wbquat = env.get_expert_attr("body_quat_seq", ind).reshape(-1, 4)
    e_wbpos = env.get_expert_attr("body_pos_seq", ind).reshape(-1, 3)

    # pose reward
    pose_diff = cur_qpos[6:] - e_qpos[6:]
    pose_dist = np.abs(pose_diff).mean()
    pose_reward = math.exp(-k_p * pose_dist)

    # global quat reward
    wpose_diff = multi_quat_norm_v2(multi_quat_diff(cur_wbquat.flatten(), e_wbquat.flatten()))
    wpose_diff = np.abs(wpose_diff).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity regularization reward
    vel_dist = np.abs(cur_qvel).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    jpos_dist = np.linalg.norm(diff, axis=1).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # object position reward
    epose_obj = env.get_expert_attr("obj_pose_seq", ind)
    obj_pose = env.get_obj_qpos()
    obj_diff = obj_pose[:3] - epose_obj[:3]
    obj_dist = np.linalg.norm(obj_diff)
    obj_pos_reward = math.exp(-k_op * obj_dist)

    # object rotation reward
    obj_rot_diff = multi_quat_norm_v2(multi_quat_diff(obj_pose[3:], epose_obj[3:]))
    obj_rot_dist = abs(obj_rot_diff[0])
    obj_rot_reward = math.exp(-k_or * obj_rot_dist)

    # object vel reward
    obj_vel = env.get_obj_qvel()
    evel_obj = np.concatenate([env.get_expert_attr("obj_vel_seq", ind), env.get_expert_attr("obj_angle_vel_seq", ind)])
    obj_vel_diff = obj_vel - evel_obj
    obj_vel_dist = abs(obj_vel_diff).mean()
    obj_vel_reward = math.exp(-k_ov * obj_vel_dist)

    # rfc reward
    obj_rfc_reward = 0
    if env.cc_cfg.residual_force:
        # vf_norm = np.linalg.norm(action[env.ndof: env.vf_dim]) / math.sqrt(3)
        obj_rfc_reward = math.exp(-k_orfc * env.rfc_score)
    else:
        obj_rfc_reward = 1.0

    hand_reward = w_p * pose_reward + w_wp * wpose_reward + w_j * jpos_reward + w_v * vel_reward
    obj_reward = w_op * obj_pos_reward + w_or * obj_rot_reward + w_ov * obj_vel_reward + w_orfc * obj_rfc_reward
    hand_reward /= (w_p + w_wp + w_j + w_v)
    obj_reward /= (w_op + w_or + w_ov + w_orfc)

    # overall reward
    reward = hand_reward * obj_reward

    # if info["end"]:
    #     reward += 1.0 / (1 - env.cc_cfg.gamma) - 1

    return reward, np.array(
        [pose_reward, wpose_reward, jpos_reward, vel_reward, obj_pos_reward, obj_rot_reward, obj_vel_reward,
         obj_rfc_reward, 1.0]
    )


reward_list = [ho_mimic_reward,
               ho_mimic_reward_2,
               ho_mimic_reward_3,
               ho_mimic_reward_4,
               ho_mimic_reward_5,
               ho_mimic_reward_6,
               ho_mimic_reward_7,
               ho_mimic_reward_8,
               ho_mimic_reward_9,
               ho_mimic_reward_10,
               ho_mimic_reward_11,
               ho_mimic_reward_12]
