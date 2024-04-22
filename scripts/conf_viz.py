import os
import time
from collections import defaultdict

import tqdm
from manopth.manolayer import ManoLayer
import torch
import numpy as np
import trimesh
import pyrender
import json
import matplotlib.pyplot as plt
import cv2

color_arr = np.array([
    [255, 255, 255, 255],
    [255, 204, 204, 255], [255, 153, 153, 255], [255, 102, 102, 255], [255, 51, 51, 255],
    [204, 255, 204, 255], [153, 255, 153, 255], [102, 255, 102, 255], [51, 255, 51, 255],
    [204, 204, 255, 255], [153, 153, 255, 255], [102, 102, 255, 255], [51, 51, 255, 255],
    [204, 255, 255, 255], [153, 255, 255, 255], [102, 255, 255, 255], [51, 255, 255, 255],
    [255, 255, 204, 255], [255, 255, 153, 255], [255, 255, 102, 255], [255, 255, 51, 255],
])

body_idx_map = np.array([
    8, 7, 6, 5,
    12, 11, 10, 9,
    20, 19, 18, 17,
    16, 15, 14, 13,
    4, 3, 2, 1
])

b = np.array([
    0,
    2.2e-2, 2.2e-2, 2.2e-2, 2.2e-2,
    1.8e-2, 1.8e-2, 1.8e-2, 1.8e-2,
    1.8e-2, 1.8e-2, 1.8e-2, 1.8e-2,
    1.8e-2, 1.8e-2, 1.8e-2, 1.8e-2,
    1.4e-2, 1.4e-2, 1.4e-2, 1.4e-2,
])

body_to_v = defaultdict(list)

is_save = True


def read_conf(fn):
    root = json.load(open(fn))
    conf_arr = [frame['conf'] for frame in root['motion_info']]
    return np.array(conf_arr)


def main():
    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True)

    hand_v, hand_j = mano_layer(th_pose_coeffs=torch.zeros((1, 48)),
                                th_betas=torch.zeros((1, 10)))

    hand_v = hand_v[0].numpy() / 1000
    hand_j = hand_j[0].numpy() / 1000
    hand_f = mano_layer.th_faces
    nv, _ = hand_v.shape

    h_mesh_t = trimesh.Trimesh(hand_v, faces=hand_f)
    for i in range(nv):
        dist = np.linalg.norm(hand_j - hand_v[i], axis=-1)
        nst_j = np.argmin(dist)

        if dist[nst_j] < b[nst_j]:
            body_to_v[nst_j].append(i)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

    h_mesh = pyrender.Mesh.from_trimesh(h_mesh_t, smooth=False)
    scene.add(h_mesh, name='hand')
    camera = pyrender.PerspectiveCamera(yfov=np.pi * 0.3, aspectRatio=1.333333)
    camera_pose = np.array([
        [0.0, -1.0, 0.0, 0.025],
        [0.0, 0.0, -1.0, -0.2],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    sl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(sl, pose=camera_pose)

    colormap = plt.get_cmap('viridis')

    # load confidence data
    seq_num = 38
    for seq_idx in tqdm.tqdm(range(1, seq_num + 1)):
        motion_data_fn = "sample_data/SingleDepth/Box/Seq" + str(seq_idx) + "/motion_seq.json"
        conf_arr = read_conf(motion_data_fn)
        frame_num, conf_num = conf_arr.shape

        if not is_save:
            # Live render
            t = 0

            def step(viewer):
                nonlocal t
                t += 1
                print(t % frame_num)

            r = pyrender.Viewer(scene,
                                viewport_size=(640, 480),
                                run_in_thread=True,
                                registered_keys={' ': step})

            while True:
                frame_id = t % frame_num
                for i in range(conf_num):
                    conf = conf_arr[frame_id][i]
                    body_idx = body_idx_map[i]
                    h_mesh_t.visual.vertex_colors[body_to_v[body_idx], :] = np.array(colormap(conf)) * 255

                h_mesh = pyrender.Mesh.from_trimesh(h_mesh_t, smooth=False)

                r.render_lock.acquire()
                for node in scene.get_nodes():
                    if node.name is None:
                        continue
                    elif node.name == 'hand':
                        scene.remove_node(node)
                scene.add(h_mesh, 'hand')
                r.render_lock.release()
        else:
            r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
            output_dir = 'results/conf/Seq' + str(seq_idx)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for fid in range(frame_num):
                for i in range(conf_num):
                    conf = conf_arr[fid][i]
                    body_idx = body_idx_map[i]
                    h_mesh_t.visual.vertex_colors[body_to_v[body_idx], :] = np.array(colormap(conf)) * 255
                h_mesh = pyrender.Mesh.from_trimesh(h_mesh_t, smooth=False)
                for node in scene.get_nodes():
                    if node.name is None:
                        continue
                    elif node.name == 'hand':
                        scene.remove_node(node)
                scene.add(h_mesh, 'hand')
                color, _ = r.render(scene)
                cv2.imwrite(output_dir + '/' + "{:04d}".format(fid) + '.png', color[:, :, [2, 1, 0]])


if __name__ == '__main__':
    main()
