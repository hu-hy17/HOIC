# -*- coding: utf-8 -*-
# @Time        : 2022/3/3 15:04
# @Author      : HuHaoyu
# @Institution : Tsinghua University
# @Email       : hhythu17@163.com
# @File        : compose.py
# @Description :
import math

import cv2
import numpy as np
import tqdm
import yaml

object_name = 'Bottle'
meta_fn = 'sample_data/SingleDepth/{}/meta.yaml'.format(object_name)
is_clear_bg = False
is_blank = False


def get_img(frame_id):
    fid = "%04d" % frame_id
    ret = []
    for i, path in enumerate(img_path_list):
        if i == 0 or i == 4:
            color_fid = "%04d" % (frame_id + color_offset)
            img = cv2.imread(path + color_fid + img_file_format)
            img = cv2.resize(img, img_size)
        elif i > 4:
            render_fid = "%04d" % (frame_id - 1)
            img = cv2.imread(path + render_fid + img_file_format)
            img = cv2.resize(img, img_size)
        else:
            img = cv2.imread(path + fid + img_file_format)
            img = cv2.resize(img, img_size)
        ret.append(img)
    return ret


def clear_bg(img):
    s = np.sum(img, axis=-1)
    pos = np.where(s == 0)
    img[pos] = np.array([255, 255, 255])
    return img


def clear_bg2(img, color):
    b_mask = g_mask = r_mask = np.zeros((img_size[1], img_size[0]))
    b_mask[np.where(img[:, :, 0] == color[0])] = 1
    g_mask[np.where(img[:, :, 1] == color[1])] = 1
    r_mask[np.where(img[:, :, 2] == color[1])] = 1
    img[np.where((b_mask == 1)&(g_mask == 1)&(r_mask == 1))] = np.array([255, 255, 255])
    return img


def compose_single_frame(img_list, frame_id):
    if is_clear_bg:
        img_list[0] = clear_bg2(img_list[0], np.array([128, 0, 0]))
        img_list[0] = cv2.resize(img_list[0], img_size)

    bg = np.zeros(bg_size, np.uint8)
    bg.fill(255)

    top = [top_margin + i * img_size[1] + 2 * i * text_margin for i in range(row_num)]
    bot = [top[i] + img_size[1] for i in range(row_num)]
    left = [lr_margin + i * img_size[0] + (i - 1) * x_margin for i in range(n_per_row)]
    right = [left[i] + img_size[0] for i in range(n_per_row)]

    for i in range(0, img_num):
        # if i == 2:
        #     bg[top:bot, left[i]:right[i]] = cv2.addWeighted(img_list[0], 0.5, img_list[1], 0.5, gamma=0)
        # else:
        ri = i // n_per_row
        ci = i % n_per_row
        bg[top[ri]:bot[ri], left[ci]:right[ci]] = img_list[i]

    # add text
    if not is_blank:
        for i in range(0, img_num):
            ri = i // n_per_row
            ci = i % n_per_row
            text_center_x = left[ci] + 320
            text_y = bot[ri] + text_margin

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3

            textsize = cv2.getTextSize(desc[i], font, font_scale, thickness)[0]
            text_width = textsize[0]
            text_x = text_center_x - text_width // 2
            cv2.putText(bg, desc[i], (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # add frame id
    # if not is_blank:
    #     frame_id_pos = (bg_size[1] // 2, bg_size[0] - bot_margin)
    #     cv2.putText(bg, str(frame_id), frame_id_pos, cv2.QT_FONT_NORMAL, 0.6, (0, 0, 0), 1)

    # cv2.imshow('debug', bg)
    # cv2.waitKey()

    return bg


if __name__ == "__main__":
    # load sequence meta data
    meta_info = yaml.safe_load(open(meta_fn, 'r'))
    obj_name = meta_info['obj_name']
    obj_name = obj_name.capitalize()
    for seq_idx, mi in enumerate(tqdm.tqdm(meta_info['test'][:1])):
        seq_name = mi[0]
        start_idx = mi[1]
        end_idx = mi[2]
        color_offset = mi[3]
        raw_seq_idx = int(seq_name[4:])
        img_path_list = [
            'J:/code/HandObjInteraction/InteractionRecon_New/data/RLData/{}/Test/{}_test_'.format(object_name, object_name) + str(raw_seq_idx) + '/617204007612_color_frame',
            'J:/code/HandObjInteraction/UIC/results/img/{}_live3/Ours/'.format(object_name) + str(seq_idx) + '/ref_',
            'J:/code/HandObjInteraction/UIC/results/img/{}_live3/SA22/'.format(object_name) + str(seq_idx) + '/ref_',
            'J:/code/HandObjInteraction/UIC/results/img/{}_live3/Ours/'.format(object_name) + str(seq_idx) + '/mimic_',
        ]
        video_name = '{}_live3_'.format(object_name) + str(seq_idx)
        # desc = ['', '', '', '', 'Color', 'Kinematic', '[Hu et al.2022]', 'Ours']
        desc = ['Color', 'Kinematic', '[Hu et al.2022]', 'Ours']

        img_num = len(img_path_list)
        img_file_format = '.png'
        img_size = (640, 480)
        lr_margin = int(0.2 * img_size[0])
        x_margin = int(0.1 * img_size[0])
        top_margin = int(0.2 * img_size[1])  # 320
        bot_margin = int(0.2 * img_size[1])  # 100
        text_margin = int(0.2 * img_size[1])  # 60

        n_per_row = 4
        row_num = math.ceil(img_num / n_per_row)
        bg_size = (img_size[1] * row_num + top_margin + bot_margin + (2 * row_num - 1) * text_margin,
                   img_size[0] * n_per_row + 2 * lr_margin + (n_per_row - 1) * x_margin,
                   3)

        start_frame = 3  # 402
        color_offset = color_offset - 3
        end_frame = 9999  # 1602
        fps = 30
        assert len(desc) == img_num

        output_path = 'results/video/'
        output_file_format = '.avi'

        stop_frame = {}

        size = (bg_size[1], bg_size[0])
        video = cv2.VideoWriter(output_path + video_name + output_file_format,
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

        try:
            total_frame = end_frame - start_frame + 1
            loading = 0.1

            for i in range(start_frame, end_frame + 1):
                imgs = get_img(i)
                frame = compose_single_frame(imgs, i)
                if i in stop_frame:
                    for t in range(0, fps * 2):
                        video.write(frame)
                else:
                    video.write(frame)
                if (i - start_frame) / total_frame > loading:
                    print(int(100 * loading))
                    loading += 0.1
            print(100)
            video.release()
        except Exception as e:
            print(e)
            video.release()
