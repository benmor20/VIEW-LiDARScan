
from typing import *
import trimesh
from threading import Thread
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def create_box(start: Sequence[float], end: Sequence[float]):
    """
    Create an array of line segments that form a box that spans from start to end

    :param start: a 3D numpy vector giving one corner of the box
    :param end: a 3D numpy vector giving the other corner of the box
    :return: a 12x2x3 numpy array giving the start and end points for the 12
        segments in the box, in no particular order
    """
    return np.array([[[start[0], start[1], start[2]], [end[0], start[1], start[2]]],
                     [[start[0], start[1], start[2]], [start[0], end[1], start[2]]],
                     [[start[0], start[1], start[2]], [start[0], start[1], end[2]]],
                     [[end[0], end[1], end[2]], [end[0], end[1], start[2]]],
                     [[end[0], end[1], end[2]], [end[0], start[1], end[2]]],
                     [[end[0], end[1], end[2]], [start[0], end[1], end[2]]],
                     [[start[0], start[1], end[2]], [start[0], end[1], end[2]]],
                     [[start[0], end[1], end[2]], [start[0], end[1], start[2]]],
                     [[start[0], end[1], start[2]], [end[0], end[1], start[2]]],
                     [[end[0], end[1], start[2]], [end[0], start[1], start[2]]],
                     [[end[0], start[1], start[2]], [end[0], start[1], end[2]]],
                     [[end[0], start[1], end[2]], [start[0], start[1], end[2]]]])


def create_triangle(corner: Sequence[float], leg1: float, leg2: float, height: float, height_axis: int) -> np.ndarray:
    """
    Create an array of line segments that forms a right triangular prism

    :param corner: a 3D vector giving the coordinates of the basis corner of the triangle
    :param leg1: a float, the length of the triangle leg on the first axis
    :param leg2: a float, the length of the trianlge leg on the second axis
    :param height: a float, the height of the prism
    :param height_axis: an int, the axis index that the height extends into
    :return: a 9x2x3 numpy array giving 9 line segments which form a triangular prism
    """
    leg1_vec = np.zeros((3,))
    leg2_vec = np.zeros((3,))
    height_vec = np.zeros((3,))
    leg1_vec[1 if height_axis == 0 else 0] = leg1
    leg2_vec[1 if height_axis == 2 else 2] = leg2
    height_vec[height_axis] = height
    corner1 = corner + leg1_vec
    corner2 = corner + leg2_vec
    corner0_up = corner + height_vec
    corner1_up = corner1 + height_vec
    corner2_up = corner2 + height_vec

    return np.array([
        [corner, corner1],
        [corner, corner2],
        [corner1, corner2],
        [corner0_up, corner1_up],
        [corner0_up, corner2_up],
        [corner1_up, corner2_up],
        [corner, corner0_up],
        [corner1, corner1_up],
        [corner2, corner2_up]
    ])


def remove_faces_from_vertex_mask(mesh: trimesh.primitives.Trimesh, mask: np.ndarray, invert: bool = False):
    """
    Remove the faces from a mesh given a mask of vertices to remove

    Since mesh.update_vertices does not properly fix the faces, use this function to first remove all faces which use
    any vertices in the mask

    :param mesh: the trimesh Mesh to remove faces from
    :param mask: a logical numpy vector, masking the vertices in mesh
    :param invert: a bool, whether to invert the given mask (to remove 1s instead of 0s)
    """
    real_mask = ~mask if invert else mask
    face_mask = real_mask[mesh.faces[:, 0]] & real_mask[mesh.faces[:, 1]] & real_mask[mesh.faces[:, 2]]
    mesh.update_faces(face_mask)


def remove_points_in_box(mesh: trimesh.primitives.Trimesh, min_box: np.ndarray, max_box: np.ndarray):
    """
    Remove all points from a mesh that are within a given box

    To remove the points, this removes the faces which contain the points

    :param mesh: the trimesh Mesh to remove points from
    :param min_box: a 3D numpy vector giving the lower bounds of the box
    :param max_box: a 3D numpy vector giving the upper bounds of the box
    """
    mask_by_dim = (mesh.vertices >= min_box) & (mesh.vertices <= max_box)
    vertex_mask = mask_by_dim[:, 0] & mask_by_dim[:, 1] & mask_by_dim[:, 2]
    remove_faces_from_vertex_mask(mesh, vertex_mask, True)


def remove_points_in_triangle(mesh: trimesh.primitives.Trimesh, corner: Sequence[float], leg1: float, leg2: float, height: float, height_axis: int):
    """
    Remove all points from a mesh that are within a given triangular prism

    To remove the points, this removes the faces which contain the points

    :param mesh: the trimesh Mesh to remove points from
    :param corner: a 3D vector giving the coordinates of the basis corner of the triangle
    :param leg1: a float, the length of the triangle leg on the first axis
    :param leg2: a float, the length of the trianlge leg on the second axis
    :param height: a float, the height of the prism
    :param height_axis: an int, the axis index that the height extends into
    """
    leg1_ax = 1 if height_axis == 0 else 0
    leg2_ax = 1 if height_axis == 2 else 2
    points2d = (mesh.vertices - np.array(corner))[:, (leg1_ax, leg2_ax)] * np.sign([leg1, leg2])
    l1 = abs(leg1)
    l2 = abs(leg2)
    vertex_mask_2d = (points2d[:, 0] >= 0)\
                     & (points2d[:, 1] >= 0)\
                     & (points2d[:, 0] <= l1)\
                     & (points2d[:, 1] <= l2)\
                     & (points2d[:, 0] * l2 + points2d[:, 1] * l1 <= l1 * l2)
    lo_height = corner[height_axis] + min(height, 0)
    hi_height = corner[height_axis] + max(height, 0)
    vertex_mask = vertex_mask_2d\
                  & (mesh.vertices[:, height_axis] >= lo_height)\
                  & (mesh.vertices[:, height_axis] <= hi_height)
    remove_faces_from_vertex_mask(mesh, vertex_mask, True)


def remove_windows(mesh: trimesh.primitives.Trimesh) -> List[trimesh.path.Path3D]:
    """
    Removes the windows from a mesh of the car

    The windows are in positions hardcoded by this function; no automatic window detection is performed

    :param mesh: the mesh to remove the windows from
    :returns: a list of trimesh Path3Ds showing the locations of each box and triangle used to remove the windows
    """
    boxes = [
        # Windshield
        (np.array([-0.7, 0.2, 0.7]), np.array([0.65, 0.6, 1.45])),
        (np.array([-0.3, 0.15, 1.2]), np.array([0.4, 0.4, 1.55])),
        # Passenger Window
        (np.array([-1.0, 0.25, 0.2]), np.array([-0.7, 0.4, 0.93])),
        (np.array([-0.85, 0.45, 0.15]), np.array([-0.55, 0.63, 0.4])),
        (np.array([-0.85, 0.23, 0.9]), np.array([-0.75, 0.3, 0.97]))
    ]

    triangles = [
        # Windshield
        ([0.35, 0.2, 1.4], 0.3, 0.15, 0.2, 1),
        ([-0.3, 0.15, 1.45], -0.3, 0.15, 0.2, 1),
        ([-0.7, 0.1, 1.4], -0.08, -0.4, 0.5, 1),
        ([0.65, 0.1, 1.4], 0.08, -0.4, 0.5, 1),
        # Passenger Window
        ([-1.0, 0.23, 0.95], -0.05, -0.75, 0.3, 0)
    ]

    for box_min, box_max in boxes:
        remove_points_in_box(mesh, box_min, box_max)
    for triangle in triangles:
        remove_points_in_triangle(mesh, *triangle)

    scene_objects = [trimesh.load_path(create_box(*box)) for box in boxes]
    scene_objects.extend(trimesh.load_path(create_triangle(*tri)) for tri in triangles)
    # scene = trimesh.Scene([mesh] + scene_objects)
    # scene.show()
    return scene_objects


def nvps_from_eye_pos(mesh: trimesh.primitives.Trimesh, eye_pos: np.ndarray):
    """
    Calculate the NVPs from a certain eye position, displaying the mesh,
        the top-down NVP graph, and saving the NVPs to a CSV

    :param mesh: the Trimesh giving the vertices and faces of the vehicle mesh
    :param eye_pos: a 3D numpy vector giving the coordinates of the eye position
    """
    # face by point by (x, y, z) (F x 3 x 3 matrix)
    faces_by_points = mesh.vertices[mesh.faces, :]

    # face by (x, y, z) (F x 3 matrix) - the highest point on each face
    high_point = np.array([max(points, key=lambda p: p[1])
                           for points in faces_by_points])

    # The directions to cast rays
    yaws = -np.deg2rad(np.arange(-20, 110))
    pitches = np.deg2rad(np.arange(-85, -5))
    yawgrid, pitchgrid = np.meshgrid(yaws, pitches)
    ray_angles = np.stack((yawgrid.flatten(), pitchgrid.flatten())).T
    # An (n x 3) matrix of unit vectors in each direction (given yaw and pitch)
    rays = np.array([
        np.sin(ray_angles[:, 0]) * np.cos(ray_angles[:, 1]),
        np.sin(ray_angles[:, 1]),
        np.cos(ray_angles[:, 0]) * np.cos(ray_angles[:, 1]),
    ]).T

    # An (n x 3) matrix where each row is equal to eye_pos
    eye_pos_repeated = np.tile(eye_pos, rays.shape[0]).reshape((-1, 3))

    print('Performing raycasts...')
    intersections = mesh.ray.intersects_first(eye_pos_repeated, rays)
    print('Done')
    print('Calculating NVP rays...')

    # For each yaw, find the tallest face that any of the rays intersect
    face_idxs_by_yaw = np.zeros((yaws.size,), dtype=int)
    for idx, yaw in enumerate(yaws):
        faces = intersections[ray_angles[:, 0] == yaw]
        face_idxs_by_yaw[idx] = int(max(faces, key=lambda f: high_point[f, 1]))

    # The height of the ground (the lowest point in the entire mesh)
    floor_y_val = np.min(mesh.vertices[:, 1])

    # A (y x 3) matrix - the direction of the NVP in a given yaw
    nvp_rays = (high_point[face_idxs_by_yaw, :] - eye_pos).T

    # Calculate the vector which points from eye position to NVP
    hypot = np.sqrt(np.sum(nvp_rays * nvp_rays, axis=0))
    nvp_rays /= hypot  # Normalize
    nvp_rays *= -(eye_pos[1] - floor_y_val) / nvp_rays[1, :] # scale until ground hit

    # Display the mesh and the NVP rays
    ray_obj = trimesh.load_path(np.array([[eye_pos, ray + eye_pos] for ray in nvp_rays.T]))
    scene = trimesh.Scene([mesh, ray_obj])
    scene.show()

    # The actual NVPs are axes 0 and 2 in NVP rays; store in CSV
    # Axis 0 is flipped relative to markerless; negate before displaying
    plt.plot(-nvp_rays[0, :], nvp_rays[2, :], 'o')
    plt.show()
    xy_vals = nvp_rays.T[:, (0, 2)] * np.array([-1, 1]) * 100 / 2.54 / 12
    with open('lidar_nvps.csv', 'w') as file:
        file.write('x (ft),y (ft)\n')
        for point in xy_vals:
            file.write(f'{point[0]},{point[1]}\n')


def main():
    src = 'data/2011HondaOdysseyScan2.glb'
    mesh: trimesh.primitives.Trimesh = trimesh.load(src, force='mesh')
    print('Loaded mesh')
    remove_windows(mesh)
    print('Filtered windows')

    dimensions = np.array([69, 68, 203]) * 0.0254  # Measurements from Google in inches
    front_left = dimensions / 2 * np.array([1, -1, 1])  # Origin is roughly center of car
    eye_pos = front_left + np.array([-0.565, 1.438, -2.292])  # Measurements from front-left of car (incl. side mirror)

    # X: Driver to passenger (positive to driver)
    # Y: Up/down (positive up)
    # Z: Front to back (positive to the front)

    nvps_from_eye_pos(mesh, eye_pos)


if __name__ == '__main__':
    main()
