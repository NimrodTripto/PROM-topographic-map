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
THRESH_CLOSED = 0.0

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
            print(f"Point {point} is close to {first_point} with thresh {thresh}")
            return False
        if np.abs(x - x2) <= thresh or np.abs(y - y2) <= thresh:
            print(f"Point {point} is close to {second_point} with thresh {thresh}")
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
                print(f"Contour {i} has father {son_contour}")
        print(f"Looping in generate_fathers_dict.")
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
        print(f"Looping in father_to_heights. Height dict: {height_dict}")
    return height_dict

def zip_contours_with_heights(contours, heights):
    return {i: (heights[i], contour) for i, contour in contours.items()}

def plot_3d_model_from_dict(contours_with_heights, diff=DIFF):
    # Initialize the plotter
    plotter = pv.Plotter()

    # Collect all points and faces
    all_points = []
    faces = []
    layers = []

    # Loop through the contours and add them to the mesh with the respective heights
    for height, contour in contours_with_heights.values():
        points = contour.squeeze()
        
        # Top height points
        z_values_top = np.full((points.shape[0], 1), height)
        points_top = np.hstack((points, z_values_top))
        
        # Bottom height points
        z_values_bottom = np.full((points.shape[0], 1), height - diff)
        points_bottom = np.hstack((points, z_values_bottom))
        
        layers.append((points_top, points_bottom))

    # Plot the planes and create the side faces
    for points_top, points_bottom in layers:
        # Add the top surface
        poly_top = pv.PolyData(points_top)
        surf_top = poly_top.delaunay_2d()
        plotter.add_mesh(surf_top, color='tan', opacity=1.0, show_edges=True)

        # Add the bottom surface
        poly_bottom = pv.PolyData(points_bottom)
        surf_bottom = poly_bottom.delaunay_2d()
        plotter.add_mesh(surf_bottom, color='tan', opacity=1.0, show_edges=True)

        # Connect the top and bottom surfaces with quads
        num_points = points_top.shape[0]
        for j in range(num_points):
            top_left = j
            top_right = (j + 1) % num_points
            bottom_left = j + num_points
            bottom_right = (j + 1) % num_points + num_points

            all_points.append(points_top[top_left])
            all_points.append(points_top[top_right])
            all_points.append(points_bottom[top_right])
            all_points.append(points_bottom[top_left])

            faces.append([4, len(all_points) - 4, len(all_points) - 3, len(all_points) - 2, len(all_points) - 1])

    # Flatten the points and faces list
    all_points = np.vstack(all_points)
    faces = np.hstack(faces)

    # Create a PolyData object
    poly = pv.PolyData()
    poly.points = all_points
    poly.faces = faces

    # Add the mesh to the plotter
    plotter.add_mesh(poly, color='tan', opacity=1.0, show_edges=True)
    
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
        contour_dict[i] = contour
    print(f"Contour dict: {contour_dict}")

    plot_all_contours(contour_dict)

    # get a dictionary of contour_index: father_index
    father_dict = generate_fathers_dict(contour_dict)
    print(f"Father dict: {father_dict}")

    # translate father_dict to a dictionary of contour_index: height
    contour_heights = father_to_heights(father_dict)  # dict of the shape {contour_number: height}

    # zip the contours with their heights
    contour_with_heights = zip_contours_with_heights(contour_dict, contour_heights)

    # draw contours with heights in 3D, using pyvista. Complete mesh between the contours
    # and plot the 3D model
    plot_3d_model_from_dict(contour_with_heights)
    print("Done")


def main():
    pass


if __name__ == "__main__":
    # Define a contour
    contour = np.array([[[0, 0]], [[0, 2]], [[2, 2]], [[2, 0]]])

    # Define a point
    point = (826, 115)

    # Use pointPolygonTest
    result = cv2.pointPolygonTest(contour, point, False)

    print(result)  # Outputs: 1.0
    