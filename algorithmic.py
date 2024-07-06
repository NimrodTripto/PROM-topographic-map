import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import Delaunay, KDTree
import matplotlib.path as mplPath
from algorithmic_advancements import calculate_curvature_from_contour,close_open_contour
import math
from scipy.interpolate import griddata, RectBivariateSpline, SmoothBivariateSpline
from scipy.spatial.distance import cdist
from algorithmic_advancements import THRESH_CLOSED
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from matplotlib.path import Path
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter
import cv2


MERGE_THRESH = 1
DIFF = 15
THRESH_EDGE = 3
JUMP_EDGE = 1
GRID_RES = 100
SIGMA = 10


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

def check_if_horrible_case(contour1, contour2, img_shape):
    # This function will check if the contours are in a horrible case: meaning,
    # containing each other.
    # We will check if point of contour1 is in contour2, and if point of contour2 is in contour1
    # (We'll use the while test to not take points that are on the edge of the image)
    flag1 = False
    flag2 = False
    index = 0
    while index < len(contour1) and is_close_to_edge(contour1[index][0], img_shape):
        index += JUMP_EDGE
    if index >= len(contour1):
        return False
    point = (float(contour1[index][0][0]), float(contour1[index][0][1]))
    # Take contour to right type
    contour2 = np.array(contour2, dtype=np.float32)
    if cv2.pointPolygonTest(contour2, point, False) == 1.0:
        flag1 = True
    index = 0
    while index < len(contour2) and is_close_to_edge(contour2[index][0], img_shape):
        index += JUMP_EDGE
    if index >= len(contour2):
        return False
    point = (float(contour2[index][0][0]), float(contour2[index][0][1]))
    # print(f"Point of contour 2 is {point}")
    # Take contour to right type
    contour1 = np.array(contour1, dtype=np.float32)
    if cv2.pointPolygonTest(contour1, point, False) == 1.0:
        flag2 = True
    return flag1 and flag2

def merge_contours(contour1, contour2, img_shape, tolerance=MERGE_THRESH):
    """
    Merge two contours into one.
    :param contour1: numpy array of shape (N, 1, 2) representing the first contour
    :param contour2: numpy array of shape (M, 1, 2) representing the second contour
    :param tolerance: distance within which points are considered common
    :return: numpy array representing the merged contour
    """

    # Get non-edge points
    non_edge_points1 = np.array([pt[0] for pt in contour1 if not is_close_to_edge(pt[0], img_shape)])
    non_edge_points2 = np.array([pt[0] for pt in contour2 if not is_close_to_edge(pt[0], img_shape)])

    # Get edge points
    edge_points1 = np.array([pt[0] for pt in contour1 if is_close_to_edge(pt[0], img_shape)])
    edge_points2 = np.array([pt[0] for pt in contour2 if is_close_to_edge(pt[0], img_shape)])

    # Find common points within tolerance
    dist_matrix = cdist(edge_points1, edge_points2)
    common_points_mask = dist_matrix < tolerance
    common_points_idx1, common_points_idx2 = np.where(common_points_mask)

    # Merge common points by averaging them
    common_points = (edge_points1[common_points_idx1] + edge_points2[common_points_idx2]) / 2

    # Combine the non-edge points and the merged common points
    merged_points = np.vstack((non_edge_points1, non_edge_points2, common_points))

    # Remove duplicates by rounding and converting to a set
    merged_points = np.unique(np.round(merged_points), axis=0)

    # Ensure the points form a valid contour by ordering them
    ordered_points = merged_points[np.argsort(merged_points[:, 0])]

    # Create the final merged contour
    final_contour = ordered_points.reshape(-1, 1, 2).astype(np.float32)

    # plot final contour as scatter
    plt.figure()
    plt.scatter(final_contour[:, 0, 0], final_contour[:, 0, 1], label='Merged Contour', s=10)
    plt.show()

    return final_contour

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

def plot_all_contours_scatter(contours):
    """
    Plot all contours in the given dictionary using scatter plots.

    Parameters:
    contours (dict): Dictionary of contours.
    """
    plt.figure()
    for i, (idx, contour) in enumerate(reversed(contours.items())):
        if not is_valid_contour_shape(contour):
            raise ValueError(f"Contour format is incorrect. Expected shape is (N, 1, 2). Got shape {contour.shape}")
        contour_points = contour.squeeze()
        plt.scatter(contour_points[:, 0], contour_points[:, 1], label=f'Contour {idx}', s=10)  # s=10 sets the marker size
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('All Contours (Scatter)')
    plt.show()

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

def is_close_to_edge(point, img_shape, thresh=THRESH_EDGE):
    """
    Check if the given point is close to the edge of the image.

    Parameters:
    point (tuple): The point to check.
    img_shape (tuple): The shape of the image.
    thresh (float): The threshold distance from the edge.

    Returns:
    bool: True if the point is close to the edge, False otherwise.
    """
    return math.fabs(point[0] - img_shape[1]) <= thresh or math.fabs(point[1] - img_shape[0]) <= thresh or point[0] <= thresh or point[1] <= thresh

def return_contained_contours(contour, all_contours, img_shape):
    contained_contours = []
    for idx, other_contour in all_contours.items():
        if other_contour is not contour:
            if not is_valid_contour_shape(other_contour):
                raise ValueError(f"Other contour format is incorrect. Expected shape is (N, 1, 2). Got shape {other_contour.shape}")

            # Ensure we don't go out of bounds
            index = 0
            while index < len(other_contour) and is_close_to_edge(other_contour[index][0], img_shape):
                index += JUMP_EDGE
            
            if index >= len(other_contour):
                continue

            # Convert the contours to the required type
            contour = np.array(contour, dtype=np.float32)
            point = (float(other_contour[index][0][0]), float(other_contour[index][0][1]))

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
            while index < len(contour) and is_close_to_edge(contour[index][0], img_shape):
                index += JUMP_EDGE

            # Convert contours to the required type
            other_contour = np.array(other_contour, dtype=np.float32)
            contour_point = (float(contour[index][0][0]), float(contour[index][0][1]))

            if cv2.pointPolygonTest(other_contour, contour_point, False) == 1.0:
                if not am_i_grandson(contour_index, i, contour_dict, img_shape):
                    return i
    return None

def am_i_grandson(contour_index_1, contour_index_2, contour_dict, img_shape):  # this function checks if contour 1 is a grandson of contour 2
    contained_in_2 = return_contained_contours(contour_dict[contour_index_2], contour_dict, img_shape)
    for contained in contained_in_2:
        contour_1 = contour_dict[contour_index_1]
        
        # Ensure the contour is of the correct shape
        if not is_valid_contour_shape(contour_1):
            raise ValueError(f"Contour format is incorrect. Expected shape is (N, 1, 2). Got shape {contour_1.shape}")

        index = 0
        while index < len(contour_1) and is_close_to_edge(contour_1[index][0], img_shape):
            index += JUMP_EDGE
        
        # Ensure we don't go out of bounds
        if index >= len(contour_1):
            continue
        
        # Convert the contours to the required type
        contained_contour = np.array(contour_dict[contained], dtype=np.int32)
        contour_point = (int(contour_1[index][0][0]), int(contour_1[index][0][1]))

        if cv2.pointPolygonTest(contained_contour, contour_point, False) == 1.0:
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

def create_continuous_pyvista_mesh(contours_with_heights, grid_res=100):
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
        np.min(all_points[:, 0]):np.max(all_points[:, 0]):complex(grid_res), 
        np.min(all_points[:, 1]):np.max(all_points[:, 1]):complex(grid_res)
    ]

    # Interpolate the heights using 'linear' interpolation method
    grid_z = griddata(all_points[:, :2], all_points[:, 2], (grid_x, grid_y), method='linear')

    # Handle NaNs in grid_z by filling with minimum height value
    min_height = np.nanmin(grid_z)
    grid_z = np.nan_to_num(grid_z, nan=min_height)

    # Apply Gaussian filter to smooth the grid_z
    grid_z = gaussian_filter(grid_z, sigma=1)

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

def draw_and_extract_contours(contour_dict, img_shape, padding=100):
    reprocessed_contours = {}

    for i, contour in contour_dict.items():
        # Ensure the contour is of the correct type and not empty
        contour = np.array(contour, dtype=np.float32)
        if contour.size == 0:
            print(f"Contour {i} is empty initially.")
            continue

        # Visualize the original contour
        # plot_contours({i: contour}, img_shape, f"Original Contour {i}")

        # Calculate contour bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Create a new blank image with sufficient padding
        new_img_shape = (h + 2 * padding, w + 2 * padding)
        img = np.zeros(new_img_shape, dtype=np.uint8)

        # Shift the contour by padding amount to ensure it fits within the new image
        shift_x = padding - x
        shift_y = padding - y
        shifted_contour = contour + [shift_x, shift_y]

        # Visualize the shifted contour
        # plot_contours({i: shifted_contour}, new_img_shape, f"Shifted Contour {i}")

        # Ensure the shifted contour has the correct shape
        if shifted_contour.size == 0 or len(shifted_contour.shape) != 3 or shifted_contour.shape[1] != 1 or shifted_contour.shape[2] != 2:
            print(f"Contour {i} has invalid shape after shifting: {shifted_contour.shape}")
            continue

        # Check for and handle invalid values in shifted_contour
        if not np.all(np.isfinite(shifted_contour)):
            print(f"Contour {i} has invalid values after shifting: {shifted_contour}")
            shifted_contour = np.nan_to_num(shifted_contour)

        # Convert to int for drawing
        shifted_contour_int = shifted_contour.astype(np.int32)

        # Draw the shifted contour on the image
        try:
            cv2.drawContours(img, [shifted_contour_int], -1, (255), thickness=cv2.FILLED)
        except cv2.error as e:
            print(f"Error drawing contour {i}: {e}")
            continue

        # Extract contours from the image
        extracted_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Ensure only one contour is found
        if len(extracted_contours) != 1:
            print(f"Expected exactly one contour to be found on the image, but found {len(extracted_contours)} for contour {i}")
            continue

        # Shift the contour back to its original position
        reprocessed_contours[i] = extracted_contours[0].astype(np.float32) - [shift_x, shift_y]

        # Visualize the reprocessed contour together with the original
        # plot_contours({i: reprocessed_contours[i], i + 1: contour}, img_shape, f"Reprocessed Contour {i}")

    return reprocessed_contours

def plot_contours_scatter(contours, img_shape, title):
    plt.figure(figsize=(10, 8))
    for i, contour in contours.items():
        plt.scatter(contour[:, 0, 0], contour[:, 0, 1], label=f"Contour {i}")
    plt.xlim([0, img_shape[1]])
    plt.ylim([0, img_shape[0]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_contours(contours, img_shape, title):
    plt.figure(figsize=(10, 8))
    for i, contour in contours.items():
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], label=f"Contour {i}")
    plt.xlim([0, img_shape[1]])
    plt.ylim([0, img_shape[0]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.show()


def is_valid_contour(contour):
    """Check if the contour is valid for drawing."""
    return contour is not None and len(contour) > 0 and contour.shape[0] > 2

def adjust_contour_points(contour, img_shape, margin):
    """Adjust contour points to ensure they are not too close to the image boundaries."""
    h, w = img_shape
    contour[:, 0, 0] = np.clip(contour[:, 0, 0], margin, w - margin)
    contour[:, 0, 1] = np.clip(contour[:, 0, 1], margin, h - margin)
    return contour.astype(int)

def create_heatmap_blur_mesh(contours_with_heights, img_shape, grid_res=100, sigma=SIGMA, padding=5, margin=0.1):
    # Expand the image size to handle edge cases
    padded_img_shape = (img_shape[0] + 2 * padding, img_shape[1] + 2 * padding)
    
    # Create an empty image to draw the contours with padding
    heatmap = np.zeros(padded_img_shape, dtype=np.float32)

    # Draw filled contours and accumulate heights
    for idx, (height, contour) in enumerate(contours_with_heights.values()):
        print(f"Processing contour {idx} with height {height}")
        if not is_valid_contour(contour):
            print(f"Contour {idx} is empty or None or invalid")
            continue
        
        if isinstance(contour, list):
            contour = np.array(contour, dtype=np.int32)  # Convert to numpy array if necessary
        
        print(f"Original contour shape: {contour.shape}")
        
        # Adjust contour points to ensure they are not too close to the image boundaries
        contour = adjust_contour_points(contour, img_shape, margin)
        print(f"Adjusted contour shape: {contour.shape}")
        
        if contour.ndim == 2:
            contour = contour[:, np.newaxis, :]  # Ensure it has the shape (n, 1, 2)
        
        if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
            # Offset the contour by the padding amount
            contour += padding
            mask = np.zeros(padded_img_shape, dtype=np.float32)
            print(f"Contour after padding adjustment: {contour.shape}")
            try:
                if not is_valid_contour(contour):
                    raise ValueError(f"Contour {idx} became invalid after padding adjustment.")
                cv2.drawContours(mask, [contour], -1, color=height, thickness=cv2.FILLED)
                heatmap += mask
            except cv2.error as e:
                print(f"Error drawing contour {idx}: {e}")
                continue
            except ValueError as ve:
                print(ve)
                continue
        else:
            print(f"Contour {idx} has unexpected shape: {contour.shape}")

    # Apply Gaussian blur to the heatmap
    heatmap_blurred = gaussian_filter(heatmap, sigma=sigma)

    # Crop back to the original image size
    heatmap_blurred = heatmap_blurred[padding:-padding, padding:-padding]

    # Define the grid
    x = np.linspace(0, img_shape[1], grid_res)
    y = np.linspace(0, img_shape[0], grid_res)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Interpolate heatmap values on the grid
    grid_z = cv2.resize(heatmap_blurred, (grid_res, grid_res))

    # Plot the heat map
    plt.imshow(grid_z, extent=(0, img_shape[1], 0, img_shape[0]), origin='lower', cmap='jet', alpha=0.5)
    plt.colorbar(label='Height')
    plt.title('Heat Map of Contours with Gaussian Blur')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Create the StructuredGrid for PyVista
    structured_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)

    return structured_grid

def algorithmic(contours, img_shape):
    # This function gets a list of contours in the "find contours" format, and makes the algorithmic part
    # so that it will be able, in the end, to model the map

    # save the first contour in a .txt file
    # with open("contour.txt", "w") as f:
    #     for point in contours[0]:
    #         f.write(f"{point[0][0]} {point[0][1]}\n")
    
    # pass all the closed contours to a given dictionary, while numbering the contours by some order
    contour_dict = {}
    redraw_dict = {}
    unedited = {}
    print(f"Img shape is {img_shape}")

    # for i, contour in enumerate(contours):
    #     # only add contours with more than 1 point
    #     if len(contour) > 1:
    #         if is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
    #             closed_contour_dict[i] = contour
    #         else:
    #             open_contour_dict[i] = contour
    #         all_contour_dict[i] = contour

    # for now
    # print contours
    # plot_contours({i: contour for i, contour in enumerate(contours)}, img_shape, "All Contours")
    for i, contour in enumerate(contours):
        # only add contours with more than 1 point
        if len(contour) > 1:
            unedited[i] = contour
            if not is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
                curvature = calculate_curvature_from_contour(contour, (img_shape[0] / 2, img_shape[1] / 2), i)
                # Added will be a list of [contour, added line, added line, ...]
                added = close_open_contour(contour,img_shape,curvature,2)
                # # Make added a numpy array of points only, where right now it's a list of [line, line, ...]
                added = np.array([point for line in added[1:] for point in line])
                # added = remove_non_edge_points(added, img_shape)
                contour = np.concatenate((contour, added))
                # Sort by x with numpy
                contour = contour[np.argsort(contour[:, 0, 0])]
                redraw_dict[i] = contour
            if not is_contour_closed2(contour, THRESH_CLOSED, img_shape[0], img_shape[1]):
                print("very bad")
            contour_dict[i] = contour

    # plot_contours_scatter(contour_dict, img_shape, "All Contours")
    # # merge unedited and redraw_dict
    # merged = {}
    # for i, contour in unedited.items():
    #     merged[i] = contour
    # for i, contour in redraw_dict.items():
    #     merged[i + len(unedited)] = contour
    # # plot_all_contours(unedited)
    # plot_contours_scatter(merged, img_shape, "All Contours")

    # plot_contours(contour_dict, img_shape, "All Contours")

    redraw_dict = draw_and_extract_contours(redraw_dict, img_shape)
    for i, contour in redraw_dict.items():
        contour_dict[i] = contour

    # Handle the case of contours that contain each other
    # Create a set to keep track of contours that have been merged
    to_delete = set()
    to_merge = {}
    # print(f"Indexes of contours are: {contour_dict.keys()}")

    for i, contour in contour_dict.items():
        if i in to_delete:
            continue
        for j, other_contour in contour_dict.items():
            if i != j and j not in to_delete and check_if_horrible_case(contour, other_contour, img_shape):
                print(f"Contours {i} and {j} are in a horrible case")
                merged_contour = merge_contours(contour, other_contour, img_shape)
                to_merge[i] = merged_contour
                to_delete.add(j)
                break
    
    # plot_contours(contour_dict, img_shape, "All Contours")

    # Update the dictionary with merged contours
    for i, merged_contour in to_merge.items():
        contour_dict[i] = merged_contour

    # Remove merged contours from the dictionary
    for j in to_delete:
        del contour_dict[j]

    # print(f"Indexes of contours are: {contour_dict.keys()}")

    # plot_contours_scatter(contour_dict, img_shape, "All Contours")

    # redraw the contours that were merged
    redraw_dict = draw_and_extract_contours(to_merge, img_shape)
    for i, contour in redraw_dict.items():
        contour_dict[i] = contour

    # plot_contours_scatter(contour_dict, img_shape, "All Contours")

    # print(f"Indexes of contours are: {contour_dict.keys()}")

    # print(f"Contour in number 0 is {closed_contour_dict[0]}")

    # for i, contour in enumerate(open_contour_dict):
    #     curvature = calculate_curvature_from_contour(contour, (0,0), i)
        # open_contour_dict[i] = close_open_contour(contour)
        # all_contour_dict[i] = close_open_contour(contour, curvature)
    

    # for i, contour in open_contour_dict.items():
    #     print(f"starting contour number {i}")
    #     curvature = calculate_curvature_from_contour(contour, (img_shape[0] / 2, img_shape[1] / 2), i)
    #     print(f"Curvature of contour {i} is {curvature}")

    father_dict = generate_fathers_dict(contour_dict, img_shape)

    # print(f"Father dict is {father_dict}")

    # print(f"contour is: {contour_dict[0]}")
    plot_contours(contour_dict, img_shape, "All Contours")

    # translate father_dict to a dictionary of contour_index: height
    contour_heights = father_to_heights(father_dict, contour_dict)  # dict of the shape {contour_number: height}

    # Add the edges of the image as a contour
    edge_contour = get_image_edge_contour(img_shape)
    contour_dict[-1] = edge_contour
    contour_heights[-1] = 0

    # zip the contours with their heights
    contour_with_heights = zip_contours_with_heights(contour_dict, contour_heights)

    plot_contours_scatter(contour_dict, img_shape, "All Contours")
    
    # create mesh
    mesh = create_pyvista_mesh(contour_with_heights)
    continuous_mesh = create_continuous_pyvista_mesh(contour_with_heights)
    heat_map_mesh = create_heatmap_blur_mesh(contour_with_heights, img_shape)

    # plot the mesh
    plot_gradient_mesh(mesh)
    # plot_gradient_mesh(continuous_mesh)
    plot_gradient_mesh(heat_map_mesh)

    # draw contours with heights in 3D, using pyvista. Complete mesh between the contours
    # and plot the 3D model
    # plot_3d_model_from_dict(contour_with_heights)
    print("Done")
