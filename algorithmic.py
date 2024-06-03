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

def try_find_son_contour(contour, contour_index, contour_dict, father_contours):
    contained_contours = return_contained_contours(contour, contour_dict)
    if len(contained_contours) == 1:
        return contained_contours[0]
    elif len(contained_contours) > 1:
        for c in contained_contours:
            if c not in father_contours:
                return c
    return None
    
def generate_fathers_dict(contour_dict):
    # We'll use a dictionary to store only the direct father of each contour
    father_dict = {}  # dict of the shape {contour_number: father_contour_number} 
    while len(father_dict) + 1 != len(contour_dict):
        for i, contour in contour_dict.items():
            son_contour = try_find_son_contour(contour, i, contour_dict, father_dict)
            if son_contour is not None:
                father_dict[son_contour] = i
    return father_dict

def father_to_heights(father_dict, initial_height=INITIAL_HGT):
    # We'll use a dictionary to store the height of each contour
    height_dict = {}  # dict of the shape {contour_number: height}
    # Assign heights to the top contours. We know a contour is a top contour if it is no one's father
    for i, father in father_dict.items():
        if i not in father_dict.values():
            height_dict[i] = initial_height
    while len(height_dict) != len(father_dict) + 1:  # +1 because the top contour is not in the father_dict
        for i, father in father_dict.items():
            if i in height_dict:
                height_dict[father] = height_dict[i] - DIFF
    return height_dict

def zip_contours_with_heights(contours, heights):
    return {i: (heights[i], contour) for i, contour in contours.items()}

def create_pyvista_mesh(contours_with_heights, diff=DIFF):
    """
    Create a pyvista mesh object from the contours with heights, including roofs and floors.

    Parameters:
    contours_with_heights (dict): Dictionary containing contours and their respective heights.

    Returns:
    pv.PolyData: PyVista PolyData mesh object.
    """
    all_points = []
    faces = []

    for height, contour in contours_with_heights.values():
        points = contour.squeeze()

        # Top height points
        z_values_top = np.full((points.shape[0], 1), height)
        points_top = np.hstack((points, z_values_top))

        # Bottom height points
        z_values_bottom = np.full((points.shape[0], 1), height - diff)
        points_bottom = np.hstack((points, z_values_bottom))

        num_points = points_top.shape[0]

        # Add points to all_points list and keep track of indices
        top_indices = []
        bottom_indices = []
        for pt in points_top:
            top_indices.append(len(all_points))
            all_points.append(pt)
        for pt in points_bottom:
            bottom_indices.append(len(all_points))
            all_points.append(pt)

        # Create faces for the top and bottom surfaces
        top_face = [num_points] + top_indices
        bottom_face = [num_points] + bottom_indices
        faces.append(top_face)
        faces.append(bottom_face)

        # Create faces for the vertical sides
        for i in range(num_points):
            next_i = (i + 1) % num_points
            side_face = [4, top_indices[i], top_indices[next_i], bottom_indices[next_i], bottom_indices[i]]
            faces.append(side_face)

    # Flatten the points and faces list
    all_points = np.array(all_points)
    faces = np.hstack(faces)

    # Create a PolyData object
    poly = pv.PolyData()
    poly.points = all_points
    poly.faces = faces

    return poly

def plot_gradient_mesh(mesh, colormap='viridis'):
    """
    Plot the given mesh with gradient colors, axes, and a height map.

    Parameters:
    mesh (pv.PolyData): The PyVista mesh object to plot.
    colormap (str): The colormap to use for coloring the mesh.
    """
    # Create a plotter object
    plotter = pv.Plotter()

    # Add the mesh to the plotter with a scalar field (height) and the specified colormap
    plotter.add_mesh(mesh, scalars=mesh.points[:, 2], cmap=colormap, show_edges=False)
    
    # Add axes
    plotter.show_axes()

    # Show the plot
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
        if len(contour) > 1 and is_contour_closed2(contour, X_MAX=img_shape[1], Y_MAX=img_shape[0]):
            contour_dict[i] = contour  

    plot_all_contours(contour_dict)

    # get a dictionary of contour_index: father_index
    father_dict = generate_fathers_dict(contour_dict)

    # translate father_dict to a dictionary of contour_index: height
    contour_heights = father_to_heights(father_dict)  # dict of the shape {contour_number: height}

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
