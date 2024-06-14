import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import Delaunay
import matplotlib.path as mplPath
from algorithmic_advancements import calculate_curvature_from_contour,close_open_contour
import math
from scipy.interpolate import griddata, RectBivariateSpline


DIFF = 60
THRESH = 2.0
THRESH_CLOSED = 0
THRESH_EDGE = 1
JUMP_EDGE = 0.1
GRID_RES = 100

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
    first_point, second_point = [0,0], [Y_MAX,X_MAX]
    x1, y1 = first_point[0], first_point[1]
    x2, y2 = second_point[0], second_point[1]
    counter = 0
    # Check the first condition
    for point in contour_points:
        x, y = point[0], point[1]
        if np.abs(x - x1) <= thresh or np.abs(y - y1) <= thresh:
            counter= counter+1
        if np.abs(x - x2) <= thresh or np.abs(y - y2) <= thresh:
            counter= counter+1

    return counter>500 or counter==0

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
    for i, (idx, contour) in enumerate(reversed(contours.items())):
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

def find_father_contour(contour, contour_index, contour_dict, father_contours, img_shape):
    # We'll go over all contours, if we find a contour that i'm his son and not his grandson then he is my father
    for i, other_contour in contour_dict.items():
        # Ensure that the contour is of the same shape
        if not is_valid_contour_shape(other_contour):
            raise ValueError(f"Other contour format is incorrect. Expected shape is (N, 1, 2). Got shape {other_contour.shape}")
        if i != contour_index:
            index = 0
            while math.fabs(contour[index][0][0] - img_shape[0]) <= THRESH_EDGE and math.fabs(contour[index][0][1] - img_shape[1]) <= THRESH_EDGE:
                index += JUMP_EDGE
            if cv2.pointPolygonTest(other_contour, (int(contour[index][0][0]), int(contour[index][0][1])), False) == 1.0:
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

def generate_fathers_dict(contour_dict, img_shape):
    # We'll use a dictionary to store only the direct father of each contour
    father_dict = {}  # dict of the shape {contour_number: father_contour_number}
    for i, contour in contour_dict.items():
        father_contour = find_father_contour(contour, i, contour_dict, father_dict, img_shape)
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

def create_continuous_pyvista_mesh(contours_with_heights, grid_res=GRID_RES):
    # Collect all points from contours with their corresponding heights
    all_points = []
    for height, contour in contours_with_heights.values():
        points = contour.squeeze()
        heights = np.full((points.shape[0], 1), height)
        all_points.append(np.hstack((points, heights)))

    # Combine all points into a single array
    all_points = np.vstack(all_points)
    
    # Define the grid
    grid_x, grid_y = np.mgrid[
        all_points[:, 0].min():all_points[:, 0].max():grid_res*1j, 
        all_points[:, 1].min():all_points[:, 1].max():grid_res*1j
    ]

    # Interpolate the heights
    grid_z = griddata(all_points[:, :2], all_points[:, 2], (grid_x, grid_y), method='linear')

    # Create the StructuredGrid
    structured_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)

    return structured_grid

def plot_gradient_mesh(mesh, colormap='viridis'):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=mesh.points[:, 2], cmap=colormap, show_edges=False)
    plotter.show_axes()
    plotter.show()

def get_image_edge_contour(img_shape):
    # Generate an edge contour for the image
    edge_contour = np.array([
        [[0, 0]],
        [[0, img_shape[0]]],
        [[img_shape[1], img_shape[0]]],
        [[img_shape[1], 0]]
    ])
    return edge_contour

def algorithmic(contours, img_shape):
    # This function gets a list of contours in the "find contours" format, and makes the algorithmic part
    # so that it will be able, in the end, to model the map

    # save the first contour in a .txt file
    # with open("contour.txt", "w") as f:
    #     for point in contours[0]:
    #         f.write(f"{point[0][0]} {point[0][1]}\n")
    
    # pass all the closed contours to a given dictionary, while numbering the contours by some order
    closed_contour_dict = {}
    open_contour_dict = {}
    all_contour_dict = {}

    # for i, contour in enumerate(contours):
    #     # only add contours with more than 1 point
    #     if len(contour) > 1:
    #         if is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
    #             closed_contour_dict[i] = contour
    #         else:
    #             open_contour_dict[i] = contour
    #         all_contour_dict[i] = contour

    # for now
    
    for i, contour in enumerate(contours):
        # only add contours with more than 1 point
        if len(contour) > 1:
            if not is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
                curvature = calculate_curvature_from_contour(contour, (img_shape[0] / 2, img_shape[1] / 2), i)
                print(f"curve is: {curvature}")
                contour = close_open_contour(contour,img_shape,curvature,2)
            if not is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
                print("very bad")
            closed_contour_dict[i] = contour
            all_contour_dict[i] = contour

    plot_all_contours(all_contour_dict)

    # print(f"Contour in number 0 is {closed_contour_dict[0]}")

    # for i, contour in enumerate(open_contour_dict):
    #     curvature = calculate_curvature_from_contour(contour, (0,0), i)
        # open_contour_dict[i] = close_open_contour(contour)
        # all_contour_dict[i] = close_open_contour(contour, curvature)
    

    # for i, contour in open_contour_dict.items():
    #     print(f"starting contour number {i}")
    #     curvature = calculate_curvature_from_contour(contour, (img_shape[0] / 2, img_shape[1] / 2), i)
    #     print(f"Curvature of contour {i} is {curvature}")

    father_dict = generate_fathers_dict(closed_contour_dict, img_shape)

    # translate father_dict to a dictionary of contour_index: height
    contour_heights = father_to_heights(father_dict, closed_contour_dict)  # dict of the shape {contour_number: height}

    # Add the edges of the image as a contour
    edge_contour = get_image_edge_contour(img_shape)
    closed_contour_dict[-1] = edge_contour
    contour_heights[-1] = 0

    # zip the contours with their heights
    contour_with_heights = zip_contours_with_heights(closed_contour_dict, contour_heights)
    
    # create mesh
    mesh = create_pyvista_mesh(contour_with_heights)
    continuous_mesh = create_continuous_pyvista_mesh(contour_with_heights)

    # plot the mesh
    plot_gradient_mesh(mesh)
    plot_gradient_mesh(continuous_mesh)

    # draw contours with heights in 3D, using pyvista. Complete mesh between the contours
    # and plot the 3D model
    # plot_3d_model_from_dict(contour_with_heights)
    print("Done")
