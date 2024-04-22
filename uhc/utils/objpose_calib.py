import copy
import time

import open3d as o3d
import numpy as np
import trimesh


def get_point_cloud(fn):
    verts = []
    if isinstance(fn, str):
        # load canonical object
        f = open(fn)
        lines = f.readlines()
        for line in lines:
            coord = line.split(' ')
            verts.append(coord)
        f.close()
        verts = np.array(verts).astype(float) / 1000
    elif isinstance(fn, np.ndarray):
        verts = fn

    faces = np.arange(0, verts.shape[0]).reshape(-1, 3)

    obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    obj2cano = obj_mesh.bounding_box_oriented.primitive.transform

    point_cloud_ori = o3d.geometry.PointCloud()
    point_cloud_ori.points = o3d.utility.Vector3dVector(obj_mesh.vertices)

    obj_mesh.apply_transform(np.linalg.inv(obj2cano))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(obj_mesh.vertices)

    return point_cloud, point_cloud_ori, np.linalg.inv(obj2cano)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def obj_pose_calib(ref_fn, process_fn, debug=False):
    # Read point cloud data
    pc1, pc1_ori, t1 = get_point_cloud(ref_fn)
    pc2, pc2_ori, t2 = get_point_cloud(process_fn)
    source = pc2_ori
    target = pc1

    # ICP calibration
    best_trans = None
    mse = 1e10
    init_trans_arr = np.tile(np.eye(4), (4, 1, 1))
    for i in range(3):
        init_trans_arr[i] *= -1
        init_trans_arr[i, i] *= -1
        init_trans_arr[i, 3] *= -1

    for i in range(4):
        init_trans = np.matmul(init_trans_arr[i], t2)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, 1, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        if reg_p2p.inlier_rmse < mse:
            mse = reg_p2p.inlier_rmse
            best_trans = reg_p2p.transformation

    if debug:
        print("Minial mse:")
        print(mse)
        print("Transformation is:")
        print(best_trans)
        draw_registration_result(source, target, best_trans)

    return best_trans
