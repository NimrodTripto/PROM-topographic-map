import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import sys
import random
import math
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import Delaunay
import matplotlib.path as mplPath


INITIAL_HGT = 3000
DIFF = 10
THRESH = 2.0
THRESH_CLOSED = 1.0

def is_contour_closed2(contour, thresh=THRESH_CLOSED, X_MAX=1000, Y_MAX=1000):
    """
    Check if the contour is closed based on given thresholds.

    Parameters:
    contour (np.ndarray): The contour to check.
    thresh1 (float): The threshold distance for the x or y coordinate.
    thresh2 (float): The threshold distance for the other coordinate.

    Returns:
    bool: True if the contour is closed, False otherwise.
    """
    contour_points = contour.squeeze()
    first_point, second_point = [0,0], [X_MAX,Y_MAX]
    x1, y1 = first_point[0], first_point[1]
    x2, y2 = second_point[0], second_point[1]
    
    # Check the first condition
    for point in contour_points:
        x, y = point[0], point[1]
        if np.abs(x - x1) <= thresh or np.abs(y - y1) <= thresh:
            return False
        if np.abs(x - x2) <= thresh or np.abs(y - y2) <= thresh:
            return False

    return True

def is_valid_contour_shape(array):
    """
    Check if the given array is of shape (N, 1, 2).

    Parameters:
    array (np.ndarray): The array to check.

    Returns:
    bool: True if the array is of shape (N, 1, 2), False otherwise.
    """
    if not isinstance(array, np.ndarray):
        return False
    if len(array.shape) != 3:
        return False
    if array.shape[1] != 1 or array.shape[2] != 2:
        return False
    return True

def plot_all_contours(contours):
    """
    Plot all contours in the given dictionary.

    Parameters:
    contours (dict): Dictionary of contours.
    """
    plt.figure()
    for i, (idx, contour) in enumerate(contours.items()):
        if not is_valid_contour_shape(contour):
            raise ValueError(f"Contour format is incorrect. Expected shape is (N, 1, 2). Got shape {contour.shape}")
        contour_points = contour.squeeze()
        plt.plot(contour_points[:, 0], contour_points[:, 1], label=f'Contour {idx}')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('All Contours')
    plt.show()

def return_contained_contours(contour, all_contours):
    contained_contours = []
    for idx, other_contour in all_contours.items():
        if other_contour is not contour:
            if not is_valid_contour_shape(other_contour):
                raise ValueError(f"Other contour format is incorrect. Expected shape is (N, 1, 2). Got shape {other_contour.shape}")
            
            point = (int(other_contour[0][0][0]), int(other_contour[0][0][1]))

            result = cv2.pointPolygonTest(contour, point, False)

            if result == 1.0:
                contained_contours.append(idx)
    return contained_contours

def find_father_contour(contour, contour_index, contour_dict, father_contours):
    # We'll go over all contours, if we find a contour that i'm his son and not his grandson then he is my father
    for i, other_contour in contour_dict.items():
        if i != contour_index:
            if cv2.pointPolygonTest(other_contour, (int(contour[0][0][0]), int(contour[0][0][1])), False) == 1.0:
                if not am_i_grandson(contour_index, i, contour_dict):
                    return i
    return None

def am_i_grandson(contour_index_1, contour_index_2, contour_dict):  # this function checks if contour 1 is a grandson of contour 2
    contained_in_2 = return_contained_contours(contour_dict[contour_index_2], contour_dict)
    for contained in contained_in_2:
        # for each contour contained in contour 2, check if it contains contour 1
        if cv2.pointPolygonTest(contour_dict[contained], (int(contour_dict[contour_index_1][0][0][0]), int(contour_dict[contour_index_1][0][0][1])), False) == 1.0:
            return True
    return False

def generate_fathers_dict(contour_dict):
    # We'll use a dictionary to store only the direct father of each contour
    father_dict = {}  # dict of the shape {contour_number: father_contour_number}
    for i, contour in contour_dict.items():
        father_contour = find_father_contour(contour, i, contour_dict, father_dict)
        if father_contour is not None:
            father_dict[i] = father_contour
    return father_dict

def count_fathers(contour_index, father_dict):
    if contour_index not in father_dict.keys():
        return 0
    return 1 + count_fathers(father_dict[contour_index], father_dict)

def father_to_heights(father_dict, contour_dict):
    # We'll use a dictionary to store the height of each contour
    height_dict = {}  # dict of the shape {contour_number: height}
    for i in contour_dict.keys():
        height_dict[i] = DIFF * count_fathers(i, father_dict) + DIFF
    return height_dict

def zip_contours_with_heights(contours, heights):
    return {i: (heights[i], contour) for i, contour in contours.items()}

def create_pyvista_mesh(contours_with_heights, diff=DIFF):
    all_points = []
    faces = []

    for height, contour in contours_with_heights.values():
        points = contour.squeeze()

        # Create Delaunay triangulation for the top and bottom surfaces
        delaunay = Delaunay(points)
        path = mplPath.Path(points)

        top_points = np.hstack((points, np.full((points.shape[0], 1), height)))
        bottom_points = np.hstack((points, np.full((points.shape[0], 1), height - diff)))

        top_indices = range(len(all_points), len(all_points) + len(top_points))
        all_points.extend(top_points)
        bottom_indices = range(len(all_points), len(all_points) + len(bottom_points))
        all_points.extend(bottom_points)

        for simplex in delaunay.simplices:
            if path.contains_point(points[simplex].mean(axis=0)):
                faces.append([3, top_indices[simplex[0]], top_indices[simplex[1]], top_indices[simplex[2]]])
                faces.append([3, bottom_indices[simplex[0]], bottom_indices[simplex[2]], bottom_indices[simplex[1]]])

        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            faces.append([4, top_indices[i], top_indices[next_i], bottom_indices[next_i], bottom_indices[i]])

    all_points = np.array(all_points)
    faces_flat = np.hstack([np.array(face) for face in faces])

    poly = pv.PolyData()
    poly.points = all_points
    poly.faces = faces_flat

    return poly

def plot_gradient_mesh(mesh, colormap='viridis'):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=mesh.points[:, 2], cmap=colormap, show_edges=False)
    plotter.show_axes()
    plotter.show()

def algorithmic(contours, img_shape):
    # This function gets a list of contours in the "find contours" format, and makes the algorithmic part
    # so that it will be able, in the end, to model the map

    # save the first contour in a .txt file
    # with open("contour.txt", "w") as f:
    #     for point in contours[0]:
    #         f.write(f"{point[0][0]} {point[0][1]}\n")
    
    # pass all the closed contours to a given dictionary, while numbering the contours by some order
    contour_dict = {}
    for i, contour in enumerate(contours):
        # only add contours with more than 1 point
        if len(contour) > 1:
            contour_dict[i] = contour  

    # get a dictionary of contour_index: father_index
    father_dict = generate_fathers_dict(contour_dict)

    plot_all_contours(contour_dict)

    # translate father_dict to a dictionary of contour_index: height
    contour_heights = father_to_heights(father_dict, contour_dict)  # dict of the shape {contour_number: height}

    # zip the contours with their heights
    contour_with_heights = zip_contours_with_heights(contour_dict, contour_heights)
    
    # create mesh
    mesh = create_pyvista_mesh(contour_with_heights)

    # plot the mesh
    plot_gradient_mesh(mesh)

    # draw contours with heights in 3D, using pyvista. Complete mesh between the contours
    # and plot the 3D model
    # plot_3d_model_from_dict(contour_with_heights)
    print("Done")
