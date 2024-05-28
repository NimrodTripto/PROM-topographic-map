import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import sys
import random


INITIAL_HGT = 3000
DIFF = 10


# def does_contain_other_contour(contours, contour):
#     # this function will check if the given contour contains any other contour from the list of contours
#     # it will return True if the contour contains any other contour, and False otherwise
#     for other_contour in contours:
#         if other_contour is not contour:
#             if cv2.pointPolygonTest(contour, tuple(other_contour[0][0]), False) == 1:
#                 return True
#     return False


def is_contour_closed(contour, threshold=1.0):
    # Calculate the distance between the first and last points
    start_point = contour[0][0]
    end_point = contour[-1][0]
    distance = np.linalg.norm(start_point - end_point)
    return distance < threshold


def calculate_heights(contours, initial_height=INITIAL_HGT):
    # This function takes a dictionary of contours and assigns heights based on the number of closed contours each contains
    def count_contained_contours(contour, all_contours):
        # Count the number of closed contours contained within the given contour
        count = 0
        for other_contour in all_contours.values():
            if other_contour is not contour and cv2.pointPolygonTest(contour, tuple(other_contour[0][0]), False) == 1:
                count += 1
        return count
    
    contour_with_heights = {}
    for i, contour in contours.items():
        if is_contour_closed(contour):
            num_contained_contours = count_contained_contours(contour, contours)
            height = initial_height - DIFF * num_contained_contours
            contour_with_heights[i] = (height, contour)
    
    return contour_with_heights

# def find_smallest_closed_contours(contours):
#     # this will find the "peaks" of the map, the smallest closed contours from the dictionary of contours
#     # they have the smallest area and they don't contain any other contour
#     # the function will return a dictionary with the (height, smallest contours) where the height of a peak is INITIAL_HGT, 
#     # while using the contour number (from the given dictionary) as the key
#     smallest_contours = {}
#     for i, contour in contours.items():
#         if is_contour_closed(contour):
#             # check if the contour doesn't contain any other contour
#             if not does_contain_other_contour(contours, contour):
#                 smallest_contours[i] = (INITIAL_HGT, contour)
#     return smallest_contours
  

def plot_3d_model_from_dict(contours_with_heights):
    # Initialize the plotter
    plotter = pv.Plotter()

    # Loop through the contours and add them to the mesh with the respective heights
    for height, contour in contours_with_heights.values():
        # Extract the points from the contour
        points = contour.squeeze()
        z_values = np.full((points.shape[0], 1), height)
        points_3d = np.hstack((points, z_values))
        
        # Create a PolyData object for the current contour
        poly = pv.PolyData(points_3d)
        
        # Create a surface from the points
        poly["scalars"] = np.arange(points_3d.shape[0])
        surf = poly.delaunay_2d()

        # Add the surface to the plotter
        plotter.add_mesh(surf, color='tan', opacity=0.6, show_edges=True)
    
    # Show the plot
    plotter.show()


def algorithmic(contours):
    # This function gets a list of contours in the "find contours" format, and makes the algorithmic part
    # so that it will be able, in the end, to model the map

    # pass all the closed contours to a given dictionary, while numbering the contours by some order
    closed_contours = {}
    for i, contour in enumerate(contours):
        closed_contours[i] = contour
    
    # get a dictionary of contours with heights for closed contours
    contour_with_heights = calculate_heights(closed_contours) # dict format is {contour_number: (height, contour)}

    # print the dictionary of contours with heights
    print(contour_with_heights)

    # draw contours with heights in 3D, using pyvista. Complete mesh between the contours
    # and plot the 3D model
    plot_3d_model_from_dict(contour_with_heights)


# Function to draw contours
def draw_contours(contours, img_size=(400, 400)):
    img = np.ones(img_size, dtype=np.uint8) * 255
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)
    return img


def main():
    contours_list = [
        np.array([[[150, 250]], [[140, 240]], [[130, 230]], [[120, 220]], [[110, 210]], [[100, 200]], [[110, 190]], [[120, 180]], [[130, 170]], [[140, 160]], [[150, 150]], [[160, 160]], [[170, 170]], [[180, 180]], [[190, 190]], [[200, 200]], [[190, 210]], [[180, 220]], [[170, 230]], [[160, 240]]], dtype=np.int32),
        np.array([[[160, 240]], [[150, 230]], [[140, 220]], [[130, 210]], [[120, 200]], [[130, 190]], [[140, 180]], [[150, 170]], [[160, 160]], [[170, 170]], [[180, 180]], [[190, 190]], [[180, 200]], [[170, 210]], [[160, 220]]], dtype=np.int32),
        np.array([[[170, 230]], [[160, 220]], [[150, 210]], [[140, 200]], [[150, 190]], [[160, 180]], [[170, 170]], [[180, 180]], [[190, 190]], [[180, 200]]], dtype=np.int32),
        np.array([[[180, 220]], [[170, 210]], [[160, 200]], [[150, 190]], [[160, 180]], [[170, 170]], [[180, 160]], [[190, 170]], [[200, 180]], [[190, 190]]], dtype=np.int32),
        np.array([[[190, 210]], [[180, 200]], [[170, 190]], [[160, 180]], [[170, 170]], [[180, 160]], [[190, 150]], [[200, 160]], [[210, 170]], [[200, 180]]], dtype=np.int32),
        np.array([[[200, 200]], [[190, 190]], [[180, 180]], [[170, 170]], [[180, 160]], [[190, 150]], [[200, 140]], [[210, 150]], [[220, 160]], [[210, 170]]], dtype=np.int32),
        np.array([[[210, 190]], [[200, 180]], [[190, 170]], [[180, 160]], [[190, 150]], [[200, 140]], [[210, 130]], [[220, 140]], [[230, 150]], [[220, 160]]], dtype=np.int32),
        np.array([[[220, 180]], [[210, 170]], [[200, 160]], [[190, 150]], [[200, 140]], [[210, 130]], [[220, 120]], [[230, 130]], [[240, 140]], [[230, 150]]], dtype=np.int32),
        np.array([[[230, 170]], [[220, 160]], [[210, 150]], [[200, 140]], [[210, 130]], [[220, 120]], [[230, 110]], [[240, 120]], [[250, 130]], [[240, 140]]], dtype=np.int32),
        np.array([[[240, 160]], [[230, 150]], [[220, 140]], [[210, 130]], [[220, 120]], [[230, 110]], [[240, 100]], [[250, 110]], [[260, 120]], [[250, 130]]], dtype=np.int32),
    ]

    # draw contours_simulated in 2D
    # Draw and display the contours
    img_with_contours = draw_contours(contours_list)
    plt.imshow(img_with_contours, cmap='gray')
    plt.title('Simulated Contours')
    plt.show()

    # run the algorithmic part
    algorithmic(contours_list)

if __name__ == '__main__':
    main()
    