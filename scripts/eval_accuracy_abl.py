import json
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

obj_name = 'Box7'
view_dir = 'front'

color_offset_dict = {
    'Box7': 396 - 3,
    'Box8': 729 - 3,
    'Bottle7': 402 - 3,
    'Bottle8': 687 - 3,
    'Banana5': 535 - 3
}

tips_map = {
    'index': 0,
    'middle': 1,
    'pinky': 2,
    'ring': 3,
    'thumb': 4
}

color_offset = color_offset_dict[obj_name]


def eval_hand_acc():
    # label_root_dir = 'F:/hhy/TechnicPaper/Label20240123/{}/front'.format(obj_name) # 'F:/hhy/TechnicPaper/Siggraph2024/Label/'
    label_root_dir = 'results/quanti/Label/Keypoints/{}'.format(obj_name)
    tps_root_dir = 'results/quanti/Data/Keypoints/'

    label_file_list = glob.glob(label_root_dir + '/{}/*.json'.format(view_dir))
    no_comp_kps = os.path.join(tps_root_dir, obj_name, 'no_surf/' + view_dir + '/pred_tps.npy')
    no_comp_kps = np.load(no_comp_kps)
    no_expl_kps = os.path.join(tps_root_dir, obj_name, 'no_explain/' + view_dir + '/pred_tps.npy')
    no_expl_kps = np.load(no_expl_kps)
    ours_kps = os.path.join(tps_root_dir, obj_name, 'Ours/' + view_dir + '/pred_tps.npy')
    ours_kps = np.load(ours_kps)

    start_idx = 0
    end_idx = len(ours_kps)

    all_pixel_err = []
    frame_idx_arr = []
    frame_to_err_map = {}
    label_count = 0

    for i, fn in enumerate(label_file_list):
        frame_id = int(fn[-9:-5])
        frame_id = frame_id - color_offset
        if frame_id <= start_idx or frame_id >= end_idx:
            continue
        labels = json.load(open(fn, 'r'))
        pixel_err = np.zeros(3)
        if len(labels['shapes']) == 0:
            continue
        for s in labels['shapes']:
            name = s['label']
            tip_idx = tips_map[name]
            gt = np.array(s['points'])
            no_comp = no_comp_kps[frame_id][tip_idx]
            no_expl = no_expl_kps[frame_id][tip_idx]
            ours = ours_kps[frame_id][tip_idx]
            pixel_err[0] += np.linalg.norm(no_comp - gt)
            pixel_err[1] += np.linalg.norm(no_expl - gt)
            pixel_err[2] += np.linalg.norm(ours - gt)

        pixel_err /= 5
        label_count += 1

        # label_count += len(labels['shapes'])
        all_pixel_err.append(pixel_err)
        frame_idx_arr.append(frame_id)
        frame_to_err_map[frame_id] = i

    all_pixel_err = np.stack(all_pixel_err)
    print(np.sum(all_pixel_err, axis=0) / label_count)
    max_err_frame = np.array(frame_idx_arr)[np.argsort(all_pixel_err[:, 2] - all_pixel_err[:, 0])[::-1]] + color_offset
    min_err_frame = np.array(frame_idx_arr)[np.argsort(all_pixel_err[:, 2] - all_pixel_err[:, 0])] + color_offset
    # dist_arr = all_pixel_err[:, 0] - all_pixel_err[:, 2]
    # frame_idx_arr = np.array(frame_idx_arr)
    # max_frame_idx = frame_idx_arr[np.argsort(dist_arr)[-10:][::-1]]
    # min_frame_idx = frame_idx_arr[np.argsort(dist_arr)[:10]]

    # plt.plot(frame_idx_arr, np.cumsum(all_pixel_err[:, 2] - all_pixel_err[:, 0]))
    # plt.show()
    # plt.plot(all_pixel_err[:, 0], label='Ref')
    # # plt.plot(all_pixel_err[:, 1], label='SA22')
    # plt.plot(all_pixel_err[:, 2], label='Ours')
    # plt.legend()
    pass


def cal_IoU(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 二值化处理，黑色区域为0，其他区域为255
    _, binary1 = cv2.threshold(image1, 1, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(image2, 1, 255, cv2.THRESH_BINARY)

    binary1 = cv2.bitwise_not(binary1)
    binary2 = cv2.bitwise_not(binary2)

    # 计算交集和并集
    intersection = cv2.bitwise_and(binary1, binary2)
    union = cv2.bitwise_or(binary1, binary2)

    # 计算相交区域和相并区域的像素数量
    intersection_pixels = cv2.countNonZero(intersection)
    union_pixels = cv2.countNonZero(union)

    # 计算IoU
    iou = intersection_pixels / union_pixels

    # cv2.imshow('img1', image1)
    # cv2.imshow('img2', image2)
    # cv2.imshow('inter', intersection)
    # cv2.imshow('union', union)
    # cv2.waitKey(30)

    return iou


def eval_obj_acc():
    no_comp_mask_root = 'results/quanti/Data/Mask/{}/no_surf/{}'.format(obj_name, view_dir)
    no_expl_mask_root = 'results/quanti/Data/Mask/{}/no_explain/{}'.format(obj_name, view_dir)
    ours_mask_root = 'results/quanti/Data/Mask/{}/Ours/{}'.format(obj_name, view_dir)
    label_root = 'results/quanti/Label/Mask/{}/{}/Masks'.format(obj_name, view_dir)
    no_comp_mask_file_list = glob.glob(no_comp_mask_root + '/*.png')
    no_expl_mask_file_list = glob.glob(no_expl_mask_root + '/*.png')
    ours_mask_file_list = glob.glob(ours_mask_root + '/*.png')
    label_file_list = glob.glob(label_root + '/*.png')
    all_IoU = []
    frame_idx_arr = []
    for fn in label_file_list:
        frame_id = int(fn[-8:-4])
        frame_id = frame_id - color_offset

        start_idx = 0
        end_idx = len(ours_mask_file_list) // 2

        if frame_id <= start_idx or frame_id >= end_idx:
            continue

        no_comp_mask_file = no_comp_mask_root + '/mimic_{:04d}.png'.format(frame_id)
        no_expl_mask_file = no_expl_mask_root + '/mimic_{:04d}.png'.format(frame_id )
        ours_mask_file = ours_mask_root + '/mimic_{:04d}.png'.format(frame_id)

        no_comp_mask = cv2.imread(no_comp_mask_file)
        no_expl_mask = cv2.imread(no_expl_mask_file)
        ours_mask = cv2.imread(ours_mask_file)

        label_mask = cv2.imread(fn)

        IoU = np.zeros(3)

        IoU[0] = cal_IoU(label_mask, no_comp_mask)
        IoU[1] = cal_IoU(label_mask, no_expl_mask)
        IoU[2] = cal_IoU(label_mask, ours_mask)

        all_IoU.append(IoU)
        frame_idx_arr.append(frame_id + color_offset)
        # cv2.imshow('gt', gt)
        # cv2.imshow('ref', ref)
        # cv2.imshow('ours', ours)
        # cv2.waitKey(30)

    all_IoU = np.stack(all_IoU)
    # max_err_frame = np.array(frame_idx_arr)[np.argsort(all_IoU[:, 0] - all_IoU[:, 2])[::-1]]
    #
    # plt.plot(frame_idx_arr, np.cumsum(all_IoU[:, 0] - all_IoU[:, 2]))
    # plt.show()

    print(np.mean(all_IoU, axis=0))


eval_hand_acc()
eval_obj_acc()
