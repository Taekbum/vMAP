import open3d as o3d
import argparse
import os
import numpy as np
import trimesh
import cv2
import yaml
from rrt import RRT
from visualize_path import visualize_path
import typing
from generate_trajectory_util import *

def normalize(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, "x must be a vector (ndim: 1)"
    return x / np.linalg.norm(x)

def look_at(
    eye,
    target: typing.Optional[typing.Any] = None,
    up: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.

    Parameters
    ----------
    eye: (3,) float
        Camera position.
    target: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).

    Returns
    -------
    T_cam2world: (4, 4) float (if return_homography is True)
        Homography transformation matrix from camera to world.
        Points are transformed like below:
            # x: camera coordinate, y: world coordinate
            y = trimesh.transforms.transform_points(x, T_cam2world)
            x = trimesh.transforms.transform_points(
                y, np.linalg.inv(T_cam2world)
            )
    """
    eye = np.asarray(eye, dtype=float)

    if target is None:
        target = np.array([0, 0, 0], dtype=float)
    else:
        target = np.asarray(target, dtype=float)

    if up is None:
        up = np.array([0, 0, -1], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), "eye must be (3,) float"
    assert target.shape == (3,), "target must be (3,) float"
    assert up.shape == (3,), "up must be (3,) float"

    # create new axes
    z_axis: np.ndarray = normalize(target - eye)
    x_axis: np.ndarray = normalize(np.cross(up, z_axis))
    y_axis: np.ndarray = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R: np.ndarray = np.vstack((x_axis, y_axis, z_axis))
    t: np.ndarray = eye

    T_cam2world: np.ndarray = compose_transform(R=R.T, t=t)
    return T_cam2world

def camera_poses_given_trajectory(room_bound, room_indices, path_list, z_min, z_max, n_points=2000):
    _random_state = np.random.RandomState(0)
    n_path = len(path_list)
    eyes = []
    targets= []
    for i in range(n_path):
        path = path_list[i]
        n_keypoints = len(path)
        
        idx = room_indices[i] # room index
        aabb = tuple([room_bound[0][idx], room_bound[1][idx]])
        # targets
        targets.append(_random_state.uniform(*aabb, (n_keypoints, 3)))

        # camera origins
        eyes_z = _random_state.uniform(z_min, z_max, (n_keypoints, 1))
        eyes.append(np.concatenate([np.array(path), eyes_z], axis=-1))
    eyes = np.concatenate(eyes, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    eyes = interpolate(eyes, n_points=n_points) 
    indices = np.linspace(0, n_points - 1, num=len(targets))
    indices = indices.round().astype(int)
    targets = sort_by(targets, key=eyes[indices])
    targets = interpolate(targets, n_points=n_points)
    
    Ts_cam2world = np.zeros((n_points, 4, 4), dtype=float)
    for i in range(n_points):
        Ts_cam2world[i] = look_at(eyes[i], targets[i])

    return Ts_cam2world

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst

if __name__ == '__main__':
    # http://www.open3d.org/docs/0.12.0/tutorial/geometry/voxelization.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./data_generation/replica_trajectory_config.yaml")
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    mesh_file = os.path.join(config['scene_file'])
    save_path = os.path.join(config['save_path'])
    os.makedirs(save_path, exist_ok=True)
    bev_file = os.path.join(save_path, 'bev_map.png')
    
    mesh = trimesh.load(mesh_file)
    mesh = trimesh_to_open3d(mesh)
    mesh_bound_max = mesh.get_max_bound()
    mesh_bound_min = mesh.get_min_bound()
    mesh_extent = mesh_bound_max - mesh_bound_min
    mesh_center = (mesh_bound_max + mesh_bound_min)/2
    scale = np.max(mesh_extent)
    center = mesh.get_center()
    mesh.scale(1 / scale, center=center)
    # o3d.visualization.draw_geometries([mesh])
    print('voxelization')
    min_bound = center + (np.array(config['min_bound'])-center) / scale
    max_bound = center + (np.array(config['max_bound'])-center) / scale
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=config['voxel_size'])
    # o3d.visualization.draw_geometries([voxel_grid])
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    
    W, H, Z = np.max(indices, 0)+1
    z_max = voxel_grid.get_voxel(max_bound)[2]
    z_min = voxel_grid.get_voxel(min_bound)[2]
    bev_img = 255*np.ones((H,W))
    n_voxel = indices.shape[0]
    for i in range(n_voxel):
        index = indices[i]
        z = index[2]
        if z > z_min and z < z_max:
            u = index[0]
            v = index[1]
            bev_img[v, u] = 0
    
    kernel_size = config['erode_kernel']
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    bev_img_erode = cv2.erode(bev_img, k)
    
    # cv2.imshow('select waypoints', bev_img)
    # cv2.waitKey(0)
    # print('generated bev image')
    
    way_points = config['way_points']
    
    print('planning trajectory')
    path_vis = []
    path_list = []
    for i in range(len(way_points)-1):
        init_state = np.copy(way_points[i])
        final_state = np.copy(way_points[i+1])
        step_size = 10
        rrt_star = RRT(bev_img_erode, init_state, final_state, step_size, max_search=1000)
        path = rrt_star.solve()
        if path is not None:
            path_list.append(path[:-1])
            path_vis.append(path)
    # visualize_path(bev_img, way_points, path_list=path_vis)
    points_list = []
    for path in path_list:
        points = []
        for pixel in path:
            point = mesh_center + scale * np.array([pixel[0]-W/2, pixel[1]-H/2, 0])/np.array([W, H, Z])
            points.append(point[:2])
        points_list.append(points)
    
    print('generate camera trajectory')
    room_bound_max = config['room_bound_max']
    room_bound_min = config['room_bound_min']
    room_bound = [room_bound_min, room_bound_max]
    room_index = config['room_index']
    poses = camera_poses_given_trajectory(room_bound, room_index, points_list, config['z_min'], config['z_max'])
    poses = poses.reshape(-1,16)
    
    os.makedirs(config['save_path'], exist_ok=True)
    np.savetxt(config['pose_file'], poses)
    
    
    