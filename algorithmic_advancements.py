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

# # types: 1 - only one face touched, 2 - two adjacent faces touched, 3 - two nonAdjacent faces touched
# def contour_type(contour, thresh=THRESH_CLOSED, X_MAX=1000, Y_MAX=1000):
#     all_faces = []
#     contour_points = contour.squeeze()
#     first_point, second_point = [0,0], [X_MAX,Y_MAX]
#     x1, y1 = first_point[0], first_point[1]
#     x2, y2 = second_point[0], second_point[1]
    
#     # Check the first condition
#     for point in contour_points:
#         x, y = point[0], point[1]
#         if np.abs(y - y1) <= thresh and 1 not in all_faces:
#             all_faces.append(1)
#         if np.abs(y - y2) <= thresh and 3 not in all_faces:
#             all_faces.append(3)
#         if np.abs(x - x1) <= thresh and 4 not in all_faces:
#             all_faces.append(4)
#         if np.abs(x - x2) <= thresh and 2 not in all_faces:
#             all_faces.append(2)
#     if (len(all_faces)==1):
#         return 1
#     elif (len(all_faces)==2):
#         if(np.abs(all_faces[0] - all_faces[1]) in [1,3]):
#             return 2
#     return 3

# def find_edges(contour, thresh=THRESH_CLOSED, X_MAX=1000, Y_MAX=1000):
#     all_edges = []
#     contour_points = contour.squeeze()
#     first_point, second_point = [0,0], [X_MAX,Y_MAX]
#     x1, y1 = first_point[0], first_point[1]
#     x2, y2 = second_point[0], second_point[1]
    
#     # Check the first condition
#     for point in contour_points:
#         x, y = point[0], point[1]
#         if np.abs(x - x1) <= thresh or np.abs(y - y1) <= thresh:
#             all_edges.append(point)
#         if np.abs(x - x2) <= thresh or np.abs(y - y2) <= thresh:
#             all_edges.append(point)
#     return all_edges
    

# def is_my_father_type_1(me, contour):
    # This function will find if the contour is the father of me
    # if the contour is of type 1 and me is of type 1
    
    # if contour_type(contour) != 1:
    #     r


def contour_to_r_theta(contour,middle_point):
    contour.squeeze()
    return [[np.abs([point[0]-middle_point[0],point[1]-middle_point[1]]),np.angle([point[0]-middle_point[0],point[1]-middle_point[1]])] for point in contour]