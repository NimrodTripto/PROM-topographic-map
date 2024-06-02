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
DIFF = 50
THRESH = 2.0
THRESH_CLOSED = 0.0


# def is_contour_closed(contour, threshold=THRESH):
#     """
#     Check if the contour is closed.

#     Parameters:
#     contour (np.ndarray): The contour to check.
#     threshold (float): The threshold to consider the contour closed.

#     Returns:
#     bool: True if the contour is closed, False otherwise.
#     """
#     # Calculate the arc length of the contour assuming it's closed
#     closed_arc_length = cv2.arcLength(contour, True)
    
#     # Calculate the arc length of the contour assuming it's open
#     open_arc_length = cv2.arcLength(contour, False)
    
#     # The contour is closed if the arc lengths are approximately equal
#     return abs(closed_arc_length - open_arc_length) < threshold

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

# def is_contour_closed3(contour, X_MAX=1000, Y_MAX=1000):
#     """
#     Check if the contour is closed based on the mean point and the furthest point.

#     Parameters:
#     contour (np.ndarray): The contour to check.
#     X_MAX (float): The maximum x-coordinate value.
#     Y_MAX (float): The maximum y-coordinate value.

#     Returns:
#     bool: True if the contour is closed, False otherwise.
#     """
#     # Ensure contour shape is (N, 1, 2)
#     contour_points = contour.squeeze()  # Shape (N, 2)

#     # Calculate the mean point of the contour
#     mean_point = np.mean(contour_points, axis=0)
#     xm, ym = mean_point

#     # Find the furthest point from the mean point
#     distances = np.linalg.norm(contour_points - mean_point, axis=1)
#     max_index = np.argmax(distances)
#     d = distances[max_index]

#     # Check the conditions for closure
#     if (xm - 0)**2 <= d**2 or (xm - X_MAX)**2 <= d**2:
#         return True
#     if (ym - 0)**2 <= d**2 or (ym - Y_MAX)**2 <= d**2:
#         return True

#     return False

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


def plot_contour_and_points(contour, points, results):
    """
    Plot the contour and the points being tested.

    Parameters:
    contour (np.ndarray): The contour to plot.
    points (list of tuples): The points being tested.
    results (list of float): The results of pointPolygonTest for each point.
    """
    plt.figure()
    contour_points = contour.squeeze()
    plt.plot(contour_points[:, 0], contour_points[:, 1], 'b-', label='Contour')
    for point, result in zip(points, results):
        color = 'r' if result == 1.0 else 'g'
        plt.plot(point[0], point[1], marker='o', color=color)
    plt.legend()
    plt.show()


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


def count_contained_contours(contour, all_contours):
        if not is_valid_contour_shape(contour):
            raise ValueError(f"Contour format is incorrect. Expected shape is (N, 1, 2). Got shape {contour.shape}")

        count = 0
        for idx, other_contour in all_contours.items():
            if other_contour is not contour:
                if not is_valid_contour_shape(other_contour):
                    raise ValueError(f"Other contour format is incorrect. Expected shape is (N, 1, 2). Got shape {other_contour.shape}")
            
                point = (int(other_contour[0][0][0]), int(other_contour[0][0][1]))
                # print(f"Testing point {point} on contour with shape {contour.shape}")

                # if idx == 0:
                #     plot_contour_and_points(contour, [point], [cv2.pointPolygonTest(contour, point, False)])

                result = cv2.pointPolygonTest(contour, point, False)
                # print(f"pointPolygonTest result for point {point}: {result}")

                if result == 1.0:
                    count += 1
        return count


def calculate_heights(contours, img_shape, initial_height=INITIAL_HGT):
    contour_with_heights = {}
    for i, contour in contours.items():
        if is_contour_closed2(contour, X_MAX=img_shape[1], Y_MAX=img_shape[0]):
            num_contained_contours = count_contained_contours(contour, contours)
            height = initial_height - DIFF * num_contained_contours # fix
            contour_with_heights[i] = (height, contour)
            print(f"Contour {i} has height {height}")
    
    return contour_with_heights



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

    plot_all_contours(contour_dict)

    # get a dictionary of contours with heights for closed contours
    contour_with_heights = calculate_heights(contour_dict, img_shape) # dict format is {contour_number: (height, contour)}

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
    