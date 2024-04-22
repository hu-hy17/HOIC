import argparse
import binascii
import cgi
import json
import struct
import os
import socketserver
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

os.add_dll_directory(os.path.abspath("mujoco210//bin"))
import sys
sys.path.append(os.getcwd())

import trimesh
from uhc.utils.objpose_calib import obj_pose_calib
from manopth.manolayer import ManoLayer
from RLTest import RLTest
from uhc.utils.transforms import matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle, \
    matrix_to_euler_angles, euler_angles_to_matrix

# os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np
from uhc.utils.config_utils.handmimic_config import Config

hand_dof_map = [0, 1, 2, 3, 4, 5,  # global
                14, 13, 15, 16,  # index
                18, 17, 19, 20,  # middle
                26, 25, 27, 28,  # little
                22, 21, 23, 24,  # ring
                10, 9, 11, 12]  # thumb

reverse_dof_idx = [
    7, 11, 15, 19, 23
]


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def build_inference_handler(handler):
    class InferenceHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.handler = handler
            super(InferenceHandler, self).__init__(*args, **kwargs)
            # self.frame_id=0

        def do_GET(self):
            page = "<form enctype=\"multipart/form-data\" method=\"post\">" \
                   "Input depth map: <input type=\"file\" name=\"depth\"><br>" \
                   "<input type=\"submit\" value=\"submit\">" \
                   "</form>"
            self.send_response(200)
            self.send_header('Content-Type',
                             'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode('utf8'))

        def do_POST(self):
            global frame_id

            time0 = time.time()
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
            )

            content_type = form['c_type'].file.read()
            if content_type == 'object':
                obj_v_raw = form['obj_v'].file.read()
                obj_v = obj_v_raw.split(' ')[:-1]
                obj_v = np.array(obj_v).astype(float).reshape(-1, 3)

                obj_n_raw = form['obj_n'].file.read()
                obj_n = obj_n_raw.split(' ')[:-1]
                obj_n = np.array(obj_n).astype(float).reshape(-1, 3)

                self.send_response(200)

                thread = threading.Thread(target=self.handler.receive_obj_info, args=(obj_v / 1000, obj_n))
                thread.start()

            elif content_type == 'pose':
                thetas_raw = form['thetas'].file.read()
                thetas = thetas_raw.split(' ')
                thetas = [float(t) for t in thetas[:-1]]

                obj_pose_raw = form['obj_pose'].file.read()
                obj_pose = obj_pose_raw.split(' ')
                obj_pose = [float(t) for t in obj_pose[:-1]]

                self.send_response(200)

                self.handler.receive_motion_info({
                    'thetas': np.array(thetas),
                    'obj_motion': np.array(obj_pose)
                })

            else:
                print("Unknown Content Type!")

    return InferenceHandler


class InferenceHandler:
    def __init__(self, cfg):
        self.rl_test = RLTest(cfg)
        self.cano_obj_fn = cfg.data_specs['cano_obj_fn']
        self.mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True)
        self.first_frame = True
        self.obj_init_pos = None
        self.obj2cano = None

    def receive_obj_info(self, obj_verts, obj_norms):
        # verts = obj_verts
        # norms = obj_norms
        # faces = np.arange(0, verts.shape[0]).reshape(-1, 3)

        obj2cano = obj_pose_calib(self.cano_obj_fn, obj_verts, debug=False)
        self.obj2cano = np.linalg.inv(obj2cano)

        # obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, process=False)
        # obj_pose_calib(self.cano_obj_fn, )
        # self.obj2cano = obj_mesh.bounding_box_oriented.primitive.transform

    def receive_motion_info(self, motion_info):
        # If object info is not processed yet, do not receive motion info
        if self.obj2cano is None:
            return

        # motion_info = {}
        thetas = torch.Tensor(motion_info['thetas']).unsqueeze(0)
        obj_pose = torch.Tensor(motion_info["obj_motion"]).view(4, 4)

        obj_pose[0:3, 3] /= 1000
        obj_pose[1:3, :] *= -1
        obj_pose = torch.matmul(obj_pose, torch.tensor(self.obj2cano).to(torch.float32))
        obj_pos = obj_pose[0:3, 3]
        obj_rot = matrix_to_quaternion(obj_pose[0:3, 0:3])

        # load motion
        hand_pose, obj_pose = self.load_motion(thetas, torch.cat([obj_pos, obj_rot], dim=0).unsqueeze(0))

        # make frame
        frame = self.rl_test.make_frame(hand_pose[0].numpy(), obj_pose[0].numpy())
        self.rl_test.add_frame(frame)

        return None

    def load_motion(self, hand_pose_seq, obj_pose_seq):
        hand_pose_seq = hand_pose_seq[:, hand_dof_map]
        hand_pose_seq[:, 0:3] = hand_pose_seq[:, 0:3] / 1000

        obj_rot_mat = quaternion_to_matrix(obj_pose_seq[:, 3:7])
        obj_pos_seq = obj_pose_seq[:, 0:3].clone()

        # left hand -> right hand
        hand_rot_mat = euler_angles_to_matrix(hand_pose_seq[:, 3:6], "XYZ")
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
        obj_pose_seq[:, 3:7] = matrix_to_quaternion(obj_rot_mat)
        hand_pose_seq[:, 3:6] = matrix_to_euler_angles(hand_rot_mat, "XYZ")

        # Translate hand and object
        if self.obj_init_pos is None:
            self.obj_init_pos = obj_pose_seq[0, 0:3].clone()
        obj_pose_seq[:, 0:3] = obj_pose_seq[:, 0:3] - self.obj_init_pos
        hand_pose_seq[:, 0:3] = hand_pose_seq[:, 0:3] - self.obj_init_pos

        # z-axis offset
        z_offset = 0.7
        obj_pose_seq[:, 2] += z_offset
        hand_pose_seq[:, 2] += z_offset

        return hand_pose_seq[:], obj_pose_seq[:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    handler = InferenceHandler(cfg)
    server = build_inference_handler(handler)
    server_address = ('127.0.0.1', 8082)
    inf_server = ThreadedHTTPServer(server_address, server)
    print('Starting server at %s, use <Ctrl-C> to stop' % (server_address[0] + ":" + str(server_address[1])))
    inf_server.serve_forever()

    # obj_fn = 'sample_data/SingleDepth/Box/Seq1/obj_n.txt'
    # obj_fv = 'sample_data/SingleDepth/Box/Seq1/obj_v.txt'
    # test_json_fn = 'sample_data/SingleDepth/Box2/Seq1/motion_seq.json'
    #
    # verts = []
    # norms = []
    # # load canonical object
    # f = open(obj_fv)
    # lines = f.readlines()
    # for line in lines:
    #     coord = line.split(' ')
    #     verts.append(coord)
    # f.close()
    # verts = np.array(verts).astype(float) / 1000
    #
    # f = open(obj_fn)
    # lines = f.readlines()
    # for line in lines:
    #     norm = line.split(' ')
    #     norms.append(norm)
    # f.close()
    # norms = np.array(norms).astype(float)
    #
    # json_root = json.load(open(test_json_fn, 'r'))
    #
    # handler.receive_obj_info(verts, norms)
    #
    # motion_seq_json = json_root['motion_info']
    # start_idx = motion_seq_json[0]['frame_id']
    # seq_len = motion_seq_json[-1]['frame_id'] - start_idx + 1
    # for i in range(seq_len):
    #     t_s = time.time()
    #     handler.receive_motion_info(motion_seq_json[i])
    #     t_s = time.time() - t_s
    #     print('Time Cost(ms): ', t_s * 1000)
