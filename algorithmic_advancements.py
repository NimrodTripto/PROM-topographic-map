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
from scipy.integrate import simps

PART_LENGTH = 50  # in pixels
BIN_SIZE = 0.01  # in radians

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


def contour_to_r_theta(contour, middle_point):
    r_theta = []
    rs,thetas = [],[]
    for point in contour:
        dx = point[0][0] - middle_point[0]
        dy = point[0][1] - middle_point[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r_theta.append([r, theta])
        rs.append(r)
        thetas.append(theta)
    # plt.figure()
    # print("rs and thetas",rs,thetas)
    # plt.plot(thetas,rs, label='Contour Segment')
    # plt.xlabel('Theta (radians)')
    # plt.ylabel('Radius (r)')
    # # plt.title(f'Contour Segment of {contour_index} in r(θ) Space')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return r_theta

def curvature(r, theta):
    # print(r[0],theta[0])
    # # Check if there are two points with same r or theta, if so then remove one of them
    # if len(r) != len(set(r)):
    #     for i in range(len(r)):
    #         for j in range(i+1, len(r)):
    #             if r[i] == r[j]:
    #                 r[j] += 0.0001
    # if len(theta) != len(set(theta)):
    #     for i in range(len(theta)):
    #         for j in range(i+1, len(theta)):
    #             if theta[i] == theta[j]:
    #                 theta[j] += 0.0001
    # print(r.shape(),theta.shape())
    r_prime = np.gradient(r, theta)
    r_double_prime = np.gradient(r_prime, theta)
    
    numerator = r**2 + 2 * r_prime**2 - r * r_double_prime
    denominator = (r_prime**2 + r**2)**(3/2)
    
    return numerator / denominator

def total_curvature(points):
    # Convert the list of points to numpy arrays
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Calculate first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Calculate second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate the curvature using the formula
    curvature = np.abs(ddy * dx - ddx * dy) / (dx**2 + dy**2)**(3/2)

    # Integrate the curvature to find the total curvature
    total_curv = simps(curvature, x)
    
    return total_curv

def divide_contour(contour, part_length=PART_LENGTH):
    # print(contour)
    parts = []
    for i in range(0, len(contour), part_length):
        parts.append(contour[i:i+part_length])
    return parts  # returns a list of lists of points

# def one_to_one_function(contour, merge_threshold=MERGE_THRESHOLD):
#     # Convert list of points to numpy array for easier manipulation
#     contour = np.array(contour)
#     r = contour[:, 0]
#     theta = contour[:, 1]

#     def bin_and_average(values, threshold):
#         sorted_values = np.sort(values)
#         binned_values = []
#         i = 0
#         while i < len(sorted_values):
#             bin_values = [sorted_values[i]]
#             while i + 1 < len(sorted_values) and sorted_values[i + 1] - sorted_values[i] <= threshold:
#                 bin_values.append(sorted_values[i + 1])
#                 i += 1
#             binned_values.append(np.mean(bin_values))
#             i += 1
#         return binned_values

#     # Bin and average r values for each theta
#     unique_theta = np.unique(theta)
#     binned_r_theta = []
#     for theta_value in unique_theta:
#         indices = np.where(theta == theta_value)[0]
#         r_values = r[indices]
#         binned_r_values = bin_and_average(r_values, merge_threshold)
#         for r_value in binned_r_values:
#             binned_r_theta.append([r_value, theta_value])

#     # Bin and average theta values for each r
#     unique_r = np.unique(r)
#     final_binned_r_theta = []
#     for r_value in unique_r:
#         indices = np.where(r == r_value)[0]
#         theta_values = theta[indices]
#         binned_theta_values = bin_and_average(theta_values, merge_threshold)
#         for theta_value in binned_theta_values:
#             final_binned_r_theta.append([r_value, theta_value])

#     # Sort the result by theta values
#     final_binned_r_theta.sort(key=lambda x: x[1])

#     return final_binned_r_theta


def one_to_one_function(contour, bin_size=BIN_SIZE):
    contour = np.array(contour)
    r = contour[:, 0]
    theta = contour[:, 1]

    # Calculate min and max theta
    min_theta = np.min(theta)
    max_theta = np.max(theta)
    # print(f"Min theta is: {min_theta}, Max theta is: {max_theta}")

    # Create bins
    bins = np.arange(min_theta, max_theta + bin_size, bin_size)
    # print(f"Bins are: {bins}")

    binned_r_theta = []
    for i in range(len(bins) - 1):
        indices = np.where((theta >= bins[i]) & (theta < bins[i + 1]))[0]
        if len(indices) > 0:
            avg_r = np.mean(r[indices])
            avg_theta = np.mean(theta[indices])
            binned_r_theta.append((avg_r, avg_theta))
    
    # print(f"Binned r-theta pairs are: {binned_r_theta}")
    return binned_r_theta


def plot_r_theta(r_theta, contour_index=0, title='Contour Segment in r(θ) Space'):
    plt.figure()
    r = [r for r, _ in r_theta]
    theta = [theta for _, theta in r_theta]
    # Check if all theta numbers, if no then print theta
    for t in theta:
        if not isinstance(t, (int, float)):
            print(f"Theta is not a number. Theta is {t}")
    plt.plot(theta, r, label='Contour Segment')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Radius (r)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_curvature_from_contour(contour, middle_point, contour_index=0):
    r_theta = contour_to_r_theta(contour, middle_point)
    # Plotting the contour segment in r(theta) space
    plot_r_theta(r_theta, contour_index, f'Contour Segment of {contour_index} in r(θ) Space')
    r_theta = one_to_one_function(r_theta)
    # Print all r values where theta is -0.8+-0.03 and print them
    # for r, theta in r_theta:
    #     if theta < -0.8 and theta > -0.86:
    #         print(f"r: {r}, theta: {theta}")
    # Plotting the contour segment in r(theta) space
    plot_r_theta(r_theta, contour_index, f'Contour Segment of {contour_index} in r(θ) Space, one to one')
    # divided = divide_contour(r_theta)
    # total_curvature = 0
    # for part in divided:
    #     if len(part) < 2:  # Skip empty or too small parts
    #         print("Skipping part")
    #         continue
    #     r, theta = zip(*part)
    #     r = np.array(r)
    #     theta = np.array(theta)
    #     total_curvature += np.sum(curvature(r, theta))
    total_curv = total_curvature(r_theta)
    # # Plotting the contour segment in r(theta) space
    # plt.figure()
    # # extrat r_list from r_theta
    # r = [r for r, _ in r_theta]
    # theta = [theta for _, theta in r_theta]
    # plt.plot(theta, r, label='Contour Segment')
    # plt.xlabel('Theta (radians)')
    # plt.ylabel('Radius (r)')
    # plt.title(f'Contour Segment of {contour_index} in r(θ) Space')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return total_curv